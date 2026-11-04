"""Adapt global ViT RoPE to image-only SAM3 input sizes, without changing weights."""
import torch


def use_configured_rpb_size(model):
    """Use the configured Python grid, avoiding CUDA scalar-to-host syncs.

    Coordinates still update through set_resolution. Reference boxes and every
    position-bias tensor are recomputed normally for each decoder layer.
    """
    decoder=model.transformer.decoder
    if getattr(decoder,'_uav_configured_rpb_size',False):
        return
    original=decoder._get_rpb_matrix
    def rpb(boxes,size):
        configured=decoder.compilable_stored_size
        return original(boxes,configured if configured is not None else size)
    decoder._get_rpb_matrix=rpb
    decoder._uav_configured_rpb_size=True


def set_resolution(model, resolution):
    if resolution <= 0 or resolution % 28:
        raise ValueError('resolution must be a positive multiple of 28')
    grid = resolution // 14
    decoder = model.transformer.decoder
    if getattr(decoder, 'boxRPB', 'none') != 'none' and decoder.compilable_stored_size != (grid, grid):
        # The upstream compiler path unconditionally uses this single-size cache.
        # Refresh it outside the compiled decoder when switching full/crop inputs.
        key = (grid, grid)
        if key not in decoder.coord_cache:
            decoder.coord_cache[key] = decoder._get_coords(grid, grid, next(decoder.parameters()).device)
        decoder.compilable_cord_cache = decoder.coord_cache[key]
        decoder.compilable_stored_size = key
    for module in model.modules():
        if not hasattr(module, 'window_size') or module.window_size != 0:
            continue
        attention = getattr(module, 'attn', None)
        if attention is None or not getattr(attention, 'use_rope', False):
            continue
        if attention.input_size == (grid, grid):
            continue
        cache = getattr(attention, '_resolution_rope_cache', {})
        fields = ['freqs_cis']
        if attention.use_rope_real:
            fields += ['freqs_cis_real', 'freqs_cis_imag']
        cache[attention.input_size] = {name: getattr(attention, name) for name in fields}
        attention._resolution_rope_cache = cache
        if (grid, grid) in cache:
            attention.input_size = (grid, grid)
            for name, tensor in cache[(grid, grid)].items():
                setattr(attention, name, tensor)
            continue
        device = attention.freqs_cis.device
        attention.input_size = (grid, grid)
        attention._setup_rope_freqs()
        attention.freqs_cis = attention.freqs_cis.to(device)
        if attention.use_rope_real:
            attention.freqs_cis_real = attention.freqs_cis_real.to(device)
            attention.freqs_cis_imag = attention.freqs_cis_imag.to(device)
