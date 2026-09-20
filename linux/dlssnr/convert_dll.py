"""CPU-only pinned dlss5 coefficient decoding (stdlib).

Strict audit extraction remains incomplete and cannot produce full-network.ok.
Explicit amd-consumer-derived mode emits experimental reconstructed tables;
it never claims missing measured maps were captured or NVIDIA equivalence.
"""
from __future__ import annotations

from array import array
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys

KNOWN_NVIDIA_SHA = 'e16bcf15e16e13f527491cdf7845b2fe6521a738d8f7c9c721866a8496e1fc8e'
UPSTREAM_COMMIT = '15799b1600d57b849597a44be53ac892b7a2faea'
ARCHIVE_OFFSET = 0x114A160
CACHE_VERSION = 'dlss5-15799b1-coefficients-v2'
DERIVED_VERSION = 'dlss5-15799b1-amd-consumer-derived-v1'
BRIDGE_HASHES = {
    'hwc-to-vit.i32':'c942210afd8ffc8a2a1e4ed81df546e5e08d4c70e20f74fdddc0fe564f224ab8',
    'vit-to-hwc.i32':'cb950400c76a6a1602ead35a817c851e552b5a301bc8e94561ed26785d1e2754',
}
DERIVED_PROVENANCE = {
    'classification':'EXPERIMENTAL reconstructed AMD-consumer layout',
    'upstream_captured':False,
    'nvidia_equivalence':False,
    'coefficients':'All learned coefficients decoded directly from SHA-verified NVIDIA DLL; no fitted or default scales.',
    'structural_zeros':'Published sparse W2 absent connections and unused ordinary C32 mix prefix only; not learned coefficients.',
    'c32':'Canonical external channels by input-projection proof; sorted W1 byte-basis hidden labels with inverse W2 permutation; AMD native QKV/P latent labels; quadrant-to-raster bias.',
    'ds4':'AMD fp8_offset with C32 canonical input and sorted first-64 W1 byte-basis output; prepare_native_front_chain.py:14-17.',
    'attention_skips':'AMD skip32; multihead compact skip conjugated by native channel low-bit swap. Not recovered NVIDIA attention skip captures.',
    'bridge':'Derived 640-token logical G/J; physical repack checked at 64/2160 and native cell mask composition; identity count 10240 corroborates. Original 640 arrays and cell NPZ unavailable.',
    'bridge_sha256':BRIDGE_HASHES,
    'secondary_source_sha256':{
        'nr-rocm/tools/inspect_packed.py':'1552b8bf2d2ec4e77ccffa83e541e2610df9a67c53332990c32dd59faa7cf988',
        'nr-rocm/tools/inspect_swin1h.py':'1fddaec6390d433afda530f26b25e6797aa6f6d9c40101773980e76b20386772',
        'nr-rocm/tools/inspect_swin2h.py':'430351e9dd025772f97efd1b57422d6b49aa77cce5449b0798751e0e812b2021',
    },
    'secondary_gpu_image_sha256':'dd38e6ede167c7a5886ae5d079022f63065b6775384d7588f185036b55877afb',
    'limitations':['Secondary manually recovered AMD address maps, not original NVIDIA connectivity proof.',
                   'Latent permutations preserve ideal dot/norm algebra, not half reduction order or vendor numerical equivalence.',
                   'Published C256 maps retain their upstream extrapolation qualification.',
                   'Runtime-ready denotes complete validated coefficient/map cache only, not GPU/full-graph/output validation.'],
}
C32_ORDER = [0,1,4,5,8,9,12,13,2,3,6,7,10,11,14,15,
             16,17,20,21,24,25,28,29,18,19,22,23,26,27,30,31]
BLOCKERS = [
    'C32 measured preblock-ffn-byte-layout/layout.npz and preblock-attention-layout/{matrix-layout,bias-layout}.npz absent; includes attention skip_channel. C32 and DS4 not emitted.',
    'C64/C128/C256 attention-layout/skip-channels.npy absent; only attention components without skip are emitted, under distinct audit names.',
    'ViT HWC bridge unresolved: requires original extent-specific forward/inverse.i32 and C512 cell mapping; 2160-token generic repack is not a 640-token HWC bridge.',
]
MULTI = {64:(0x7010,0xe0a0,0xe0b0,0xa0a0,0xf0b0),
         128:(0x18010,0x2c120,0x2c130,0x24120,0x30130),
         256:(0x58010,0x98220,0x98240,0x88220,0xa8240)}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _u32(data, offset):
    return struct.unpack_from('<I', data, offset)[0]


def _u64(data, offset):
    return struct.unpack_from('<Q', data, offset)[0]


def _f16_to_f32(h):
    return struct.unpack('<e', struct.pack('<H', h))[0]


def _f16s(payload):
    if len(payload) % 2:
        raise ValueError('odd FP16 payload length')
    return [v[0] for v in struct.iter_unpack('<e', payload)]


def _write_f32(path, values):
    data = array('f', values)
    if data.itemsize != 4:
        raise RuntimeError('32-bit float array required')
    if sys.byteorder != 'little':
        data.byteswap()
    path.write_bytes(data.tobytes())


def expected_records():
    """Pinned Development/weights-index.json: all 153 names and lengths."""
    sizes = {}
    for b in range(71):
        if 23 <= b <= 30 or 40 <= b <= 47:
            layers = [524288,263168,917568,263168]
            if b == 30:
                layers.append(524304)
        elif 31 <= b <= 38:
            layers = [4194320,4196352,3145856,2,1050624]
        else:
            ordinary = (20672 if b < 5 or b >= 66 else
                        61760 if b < 9 or b >= 62 else
                        197184 if b < 15 or b >= 56 else 689232)
            layers = [{0:21696,4:22720,8:69936,14:229936,22:820288,
                       39:525312,48:820784,56:230176,62:70048,
                       66:22784,70:21808}.get(b,ordinary)]
        for layer,size in enumerate(layers):
            sizes[f'block{b}.layer{layer}.layer'] = size
    sizes['block70.layer0.blend_scale'] = 2
    return sizes


def parse_archive(blob):
    if len(blob) < 8 or _u64(blob,0) != len(blob):
        raise RuntimeError('WEIGHTS archive size mismatch or truncated header')
    records,names = [],set()
    cursor = 8
    while cursor < len(blob):
        if cursor+16 > len(blob):
            raise RuntimeError('truncated weight record header')
        name_len = _u64(blob,cursor)
        if not 1 <= name_len <= 4096 or cursor+16+name_len > len(blob):
            raise RuntimeError('invalid weight record name length')
        try:
            name = blob[cursor+8:cursor+8+name_len].decode('ascii')
        except UnicodeDecodeError as exc:
            raise RuntimeError('non-ASCII weight record') from exc
        if name in names:
            raise RuntimeError(f'duplicate weight record {name}')
        names.add(name)
        span = _u64(blob,cursor+8+name_len)
        body = cursor+16+name_len
        end = body+span
        if span < 40 or end > len(blob):
            raise RuntimeError(f'invalid body bounds: {name}')
        size = _u64(blob,body+8)
        if _u64(blob,body) != span or size+40 != span or size % 2:
            raise RuntimeError(f'invalid payload/body span: {name}')
        if _u32(blob,body+16) != 1:
            raise RuntimeError(f'invalid dtype: {name}')
        if struct.unpack_from('<5I',blob,body+20+size) != (0,0,1,0,size//2):
            raise RuntimeError(f'invalid storage trailer: {name}')
        records.append(dict(name=name,payload_offset=body+20,payload_size=size))
        cursor = end
    expected = expected_records()
    if len(records) != 153 or names != set(expected):
        raise RuntimeError('expected exactly 153 records in the pinned record set')
    for r in records:
        if r['payload_size'] != expected[r['name']]:
            raise RuntimeError(f'wrong payload length: {r["name"]}')
    return records


def _bits(count, positions):
    return [sum(((i>>p)&1)<<b for b,p in enumerate(positions)) for i in range(count)]


def _e4m3(byte):
    sign = -1.0 if byte & 128 else 1.0
    exp,mant = (byte>>3)&15,byte&7
    if exp == 15 and mant == 7:
        return float('nan')
    return sign * (mant/512.0 if exp == 0 else (1.0+mant/8.0)*2.0**(exp-7))


FP8 = tuple(_e4m3(b) for b in range(256))


def _length(raw, expected):
    if len(raw) != expected:
        raise ValueError(f'payload length {len(raw)} != {expected}')


def _scatter_e4m3(raw, rows, cols, row_idx, col_idx):
    n = len(raw)
    if n != len(row_idx) or n != len(col_idx):
        raise ValueError('matrix payload/map length mismatch')
    # Sparse W2 in multihead FFN has only 128*C connections. The remaining
    # entries are structural zeros per native_c64_reference.py, not padding.
    mat,seen = [0.0]*(rows*cols),bytearray(rows*cols)
    for i in range(n):
        r,c = row_idx[i],col_idx[i]
        if not (0 <= r < rows and 0 <= c < cols) or seen[r*cols+c]:
            raise ValueError('invalid or duplicate matrix coordinate')
        seen[r*cols+c] = 1
        mat[r*cols+c] = FP8[raw[i]]
    return mat


def _mapped_matrix(raw, rows, cols, rb, cb):
    """Dense bit permutations; compact axis LUTs avoid huge index arrays."""
    _length(raw, rows*cols)
    if (1<<len(rb),1<<len(cb)) != (rows,cols) or sorted(rb+cb) != list(range(len(rb)+len(cb))):
        raise ValueError('matrix bit mapping is not a bijection')
    # Gather from the logical axes. Each byte is visited exactly once.
    ri = [sum(((i>>b)&1)<<p for b,p in enumerate(rb)) for i in range(rows)]
    ci = [sum(((i>>b)&1)<<p for b,p in enumerate(cb)) for i in range(cols)]
    return [FP8[raw[r|c]] for r in ri for c in ci]


def _skip_order(channels):
    return [(s//16)*16+(s%8)*2+(s%16//8) for s in range(channels)]


def _skip(raw, order):
    _length(raw,2*len(order))
    if sorted(order) != list(range(len(order))):
        raise ValueError('skip map is not a permutation')
    result = [0.0]*len(order)
    for i,value in enumerate(_f16s(raw)):
        result[order[i]] = value
    return result


def _ffn_maps(channels):
    if channels not in (64,128,256):
        raise RuntimeError('unresolved measured C32 FFN maps')
    d = channels.bit_length()-1
    group = list(range(12,d+7))
    return {
        'w1_input':_bits(4*channels*channels,[1,0,4,5,2]+group),
        'w1_hidden':_bits(4*channels*channels,[3,6,7,8,9,10,11]+list(range(d+7,2*d+2))),
        'w2_hidden':_bits(128*channels,[1,0,4,5,2,10,11]+group),
        'w2_output':_bits(128*channels,[3,6,7,8,9]+group),
        'w3_input':_bits(channels*channels,[1,0,4,5,2]+list(range(d+5,2*d))),
        'w3_output':_bits(channels*channels,[3,6,7,8,9]+list(range(10,d+5))),
    }


def _preblock(raw):
    """preblock_mix_reference.py: mix inserted after FFN weights."""
    _length(raw,21696)
    mix = [0.0]*512
    for s,value in enumerate(_f16s(raw[8208:9232])):
        channel = (s//64)*4+(s//32%2)+2*(s//4%2)
        feature = (s//8%4)*4+s%4
        mix[channel*16+feature] = value
    return mix,raw[:8208]+raw[9232:]


def _post70(raw):
    """native_post70_reference.py; inserted zeros are unused padding."""
    _length(raw,21808)
    ordinary = raw[:0x2050]+bytes(16)+raw[0x20d0:0x5130]
    scales = _skip(raw[0x2050:0x2090],C32_ORDER)+_skip(raw[0x2090:0x20d0],C32_ORDER)
    head = [0.0]*512
    rr,cc = _bits(512,[2,5,6,7]),_bits(512,[0,1,3,4,8])
    for i,value in enumerate(_f16s(raw[0x5130:])):
        head[rr[i]*32+cc[i]] = value
    return ordinary,scales,[head[r*32+c] for r in (0,2,4) for c in range(32)]


def _upsample(raw, channels):
    """native_upsample48/66_reference.py: internal matrix, not prefix."""
    if channels == 32:
        _length(raw,22784)
        body = raw[:0x2000]+raw[0x2800:0x2860]+raw[0x28a0:]
        matrix = _mapped_matrix(raw[0x2000:0x2800],32,64,[3,6,7,8,9],[1,0,4,5,2,10])
        reordered = [0.0]*2048
        for i,source in enumerate(_skip_order(32)):
            reordered[C32_ORDER[i]*64:(C32_ORDER[i]+1)*64] = matrix[source*64:(source+1)*64]
        return reordered+_skip(raw[0x2860:0x28a0],C32_ORDER),body
    size,n,begin,ffskip,qkv = {64:(70048,61760,0x7000,0x9000,0x70a0),
        128:(230176,197184,0x18000,0x20000,0x18120),
        256:(820784,689232,0x58000,0x78000,0x58220)}[channels]
    _length(raw,size)
    body = bytearray(n)
    body[:begin] = raw[:begin]
    body[begin+16:begin+16+2*channels] = raw[ffskip:ffskip+2*channels]
    body[qkv:] = raw[ffskip+4*channels:]
    d = channels.bit_length()-1
    matrix = _mapped_matrix(raw[begin:ffskip],channels,2*channels,
                           [3]+list(range(6,d+5)),[1,0,4,5,2]+list(range(d+5,2*d+1)))
    return matrix+_skip(raw[ffskip+2*channels:ffskip+4*channels],_skip_order(channels)),bytes(body)


def _downsample(raw, channels, *, layout_mode=None):
    """derive_native_ds_layout.py + prepare_native_front_chain.py."""
    derived = _experimental(layout_mode)
    if channels == 32:
        if not derived:
            raise RuntimeError('unresolved measured C32 DS map')
        _length(raw,22720)
        # Sorted first 64 W1 byte-basis labels, as prepare_native_front_chain.
        # Unlike its canonical C32 input, C64 output uses the swapped basis.
        return _mapped_matrix(raw[0x50b0:0x58b0],64,32,[3,6,7,8,9,10],[0,1,4,5,2])
    offset,size = {64:(0xf130,69936),128:(0x30230,229936),256:(0xa8440,820288)}[channels]
    _length(raw,size)
    d = channels.bit_length()-1
    return _mapped_matrix(raw[offset:offset+2*channels*channels],2*channels,channels,
                          [3,6,7,8,9]+list(range(10,d+6)),[1,0,4,5,2]+list(range(d+6,2*d+1)))


def _experimental(layout_mode):
    if layout_mode not in (None, 'amd-consumer-derived'):
        raise ValueError('unknown layout_mode; expected amd-consumer-derived or None')
    return layout_mode == 'amd-consumer-derived'


def _unpack_c32(raw, *, layout_mode=None):
    """EXPERIMENTAL AMD consumer reconstruction, not captured NVIDIA maps.

    inspect_packed.fp8_offset uses canonical C32 external channels (prefix
    proof). Sorting W1's input-zero byte representatives labels hidden units
    by swapping native low bits 0/1. Apply the same permutation to W2's input.
    Q/K/V retain AMD latent labels; P consumes those labels unchanged.
    Bias composes AMD quadrant pixels with native raster pixels. The explicit
    skip32 field applies to both AMD skips, NOT a recovered NVIDIA skip array.
    """
    if not _experimental(layout_mode):
        raise RuntimeError('unresolved measured C32 FFN/attention layout.npz and skip_channel')
    if len(raw) not in (20672,22720):
        raise ValueError('invalid C32 body size')
    w1 = _mapped_matrix(raw[:4096],128,32,[3,6,7,8,9,10,11],[0,1,4,5,2])
    w2 = _mapped_matrix(raw[4096:8192],32,128,[6,3,7,8,9],[1,0,4,5,2,10,11])
    # Published ordinary export reserves an UNUSED 512-float mix field.
    ffn = [0.0]*512+w1+w2+_skip(raw[0x2010:0x2050],C32_ORDER)
    attention = []
    for base in (0x2060,0x2460,0x2860,0x4c70):
        attention.extend(_mapped_matrix(raw[base:base+1024],32,32,[6,3,7,8,9],[0,1,4,5,2]))
    attention.extend(_bias(raw[0x2c60:0x4c60],1))
    attention.append(struct.unpack_from('<f',raw,0x4c60)[0])
    attention.extend(_skip(raw[0x5070:0x50b0],C32_ORDER))
    return ffn,attention


def _multi_length(raw, channels):
    sizes = {64:(61760,69936),128:(197184,229936),256:(689232,820288)}
    if len(raw) not in sizes[channels]:
        raise ValueError('invalid multihead body size')


def _unpack_multi(raw, channels, *, layout_mode=None):
    """derive_native_ffn_layout.py; attention skip remains unresolved."""
    derived = _experimental(layout_mode)
    _multi_length(raw,channels)
    hidden = 4*channels
    b2,b3 = hidden*channels,hidden*channels+128*channels
    m = _ffn_maps(channels)
    w1 = _scatter_e4m3(raw[:b2],hidden,channels,m['w1_hidden'],m['w1_input'])
    w2 = _scatter_e4m3(raw[b2:b3],channels,hidden,m['w2_output'],m['w2_hidden'])
    w3 = _scatter_e4m3(raw[b3:b3+channels*channels],channels,channels,m['w3_output'],m['w3_input'])
    fs = MULTI[channels][0]
    attention = None
    if derived:
        # AMD compact skip conjugated by native input-view low-bit swap.
        # Equal to published FFN order, NOT a captured NVIDIA attention map.
        at = MULTI[channels][4]
        attention = _multi_attention_components(raw,channels)+_skip(raw[at:at+2*channels],_skip_order(channels))
    return w1+w2+w3+_skip(raw[fs:fs+2*channels],_skip_order(channels)),attention


def _bias(raw, heads):
    _length(raw,heads*4096*2)
    values = _f16s(raw)
    q,k = _bits(4096,[5,6,10,7,1,11]),_bits(4096,[0,3,8,4,2,9])
    result = [0.0]*len(values)
    for i,value in enumerate(values):
        j = i%4096
        result[(i//4096)*4096+q[j]*64+k[j]] = value
    return result


def _interleaved(raw, n):
    _length(raw,3*n)
    return [b''.join(raw[i+part*1024:i+(part+1)*1024] for i in range(0,3*n,3072)) for part in range(3)]


def _multi_attention_components(raw, channels):
    """derive_native_attention_layout.py. NOT a complete attention table."""
    _multi_length(raw,channels)
    _,scale,p,bias,_ = MULTI[channels]
    base = {64:0x70a0,128:0x18120,256:0x58220}[channels]
    d,n = channels.bit_length()-1,channels*channels
    rb,cb = [3,6,7,8,9]+list(range(10,d+5)),[1,0,4,5,2]+list(range(d+5,2*d))
    result = []
    for chunk in _interleaved(raw[base:base+3*n],n):
        result.extend(_mapped_matrix(chunk,channels,channels,rb,cb))
    result.extend(_mapped_matrix(raw[p:p+n],channels,channels,rb,cb))
    result.extend(_bias(raw[bias:scale],channels//32))
    result.extend(struct.unpack_from('<'+'f'*(channels//32),raw,scale))
    return result


def _split(raws):
    """native_split_weights.py, including both independently published skips."""
    for raw,size in zip(raws,[524288,263168,917568,263168],strict=True):
        _length(raw,size)
    rb,cb = [3,6,7,8,9,10,11,12,13],[1,0,4,5,2,14,15,16,17]
    def matrix(raw):
        return _mapped_matrix(raw,512,512,rb,cb)
    fw = matrix(raws[0][:262144])
    for start,rows,cols,r,c in [(0x40000,256,64,[3,6,7,8,9,10,11,12],[1,0,4,5,2,13]),
                               (0x60000,64,256,[3,6,7,8,9,10],[1,0,4,5,2,11,12,13])]:
        for g in range(8):
            fw.extend(_mapped_matrix(raws[0][start+g*16384:start+(g+1)*16384],rows,cols,r,c))
    fp = matrix(raws[1][:262144])+_skip(raws[1][262144:],_skip_order(512))
    aw = []
    for chunk in _interleaved(raws[2][:0xc0000],262144):
        aw.extend(matrix(chunk))
    aw.extend(matrix(raws[3][:262144]))
    aw.extend(_bias(raws[2][0xc0000:0xe0000],16))
    aw.extend(struct.unpack('<16f',raws[2][0xe0000:]))
    aw.extend(_skip(raws[3][262144:],_skip_order(512)))
    return fw,fp,aw


def _vit_matrix(raw, inputs, outputs):
    """native_vit_linear_reference.py, NOT raw row-major FP8."""
    if (inputs,outputs) not in ((1024,4096),(4096,1024),(1024,1024)):
        raise ValueError('invalid ViT matrix shape')
    ib,ob = inputs.bit_length()-1,outputs.bit_length()-1
    return _mapped_matrix(raw,outputs,inputs,[6,3,9,7,8]+list(range(10,ob+5)),
                          [0,1,2,4,5]+list(range(ob+5,ib+ob)))


def _vit_expand(raw):
    _length(raw,4194320)
    if any(raw[4194304:]):
        raise ValueError('nonzero ViT expand padding')
    return _vit_matrix(raw[:4194304],1024,4096)


def _vit_residual(raw, inputs):
    _length(raw,inputs*1024+2048)
    matrix = _vit_matrix(raw[:inputs*1024],inputs,1024)
    half = _f16s(raw[inputs*1024:])
    # unpack_vit_matrices.MATRIX_OUTPUT_TO_RAW is logical -> raw (gather).
    order = [sum(((i>>b)&1)*p for b,p in enumerate([1,8,16,2,4,32,64,128,256,512])) for i in range(1024)]
    return matrix+[half[i] for i in order]


def _vit_qkv(raw):
    """native_vit_qkv_reference.py: 128-byte scale prefix, 1KiB Q/K/V."""
    _length(raw,3145856)
    result = []
    for chunk in _interleaved(raw[128:],1048576):
        result.extend(_vit_matrix(chunk,1024,1024))
    return result+list(struct.unpack('<32f',raw[:128]))


def _head(raw):
    _length(raw,524304)
    return _mapped_matrix(raw[:524288],1024,512,[3,6,7,8,9,10,11,12,13,14],[1,0,4,5,2,15,16,17,18])


def _decoder39(raw):
    _length(raw,525312)
    return _mapped_matrix(raw[:524288],512,1024,[3,6,7,8,9,10,11,12,13],[1,0,4,5,2,14,15,16,17,18])+_skip(raw[524288:],_skip_order(512))


def _derived_bridge():
    """640-token G/J from bridge-audit: derived, NOT original-640-captured.

    Composition of physical repack (checked at 64/2160), native cell masks
    and ViT logical coordinates. The inverse is independently expressed.
    """
    forward,back = array('i'),array('i')
    for t in range(640):
        p = (t&~15)|((t&1)<<3)|((t&14)>>1)
        for c in range(1024):
            h = (c&~31)|((c&1)<<1)|((c&2)>>1)|((c&4)<<2)|((c&24)>>1)
            forward.append(p*1024+h)
    for p in range(640):
        t = (p&~15)|((p&8)>>3)|((p&7)<<1)
        for h in range(1024):
            c = (h&~31)|((h&1)<<1)|((h&2)>>1)|((h&16)>>2)|((h&12)<<1)
            back.append(t*1024+c)
    return forward,back


def _multi_blocks():
    return [(b,64 if b<9 or b>=62 else 128 if b<15 or b>=56 else 256)
            for b in list(range(5,23))+list(range(48,66))]


def expected_tables(*, layout_mode=None):
    """Only independently specified outputs; audit names cannot masquerade as full tables."""
    derived = _experimental(layout_mode)
    sizes = {'block0-mix.audit.f32':512,'post70-head.f32':96,'post70-scales.f32':64,
             'head-matrix.f32':524288,'decoder39-weights.f32':524800}
    for b,ch in _multi_blocks():
        sizes[f'block{b}-ffn.f32'] = 9*ch*ch+ch
        sizes[f'block{b}-attention-components.audit.f32'] = 4*ch*ch+ch//32*4096+ch//32
    for b,ch in [(8,64),(14,128),(22,256)]:
        sizes[f'block{b}-ds.f32'] = 2*ch*ch
    for b in list(range(23,31))+list(range(40,48)):
        sizes[f'block{b}-ffwd.f32'] = 524288
        sizes[f'block{b}-ffwd-projection.f32'] = 262656
        sizes[f'block{b}-attention.f32'] = 1114640
    for b in range(31,39):
        for suffix,n in [('expand',4194304),('contract',4195328),('qkv',3145760),('projection',1049600)]:
            sizes[f'block{b}-{suffix}.f32'] = n
    for b,ch in [(48,256),(56,128),(62,64),(66,32)]:
        sizes[f'block{b}-weights.f32'] = 2*ch*ch+ch
    if derived:
        # Keep all confirmed audit files byte-for-byte; append full runtime tables.
        for b in list(range(5))+list(range(66,71)):
            prefix = 'post70' if b == 70 else f'block{b}'
            sizes[prefix+'-ffn.f32'] = 8736
            sizes[prefix+'-attention.f32'] = 8225
        sizes['block4-ds.f32'] = 2048
        for b,ch in _multi_blocks():
            sizes[f'block{b}-attention.f32'] = 4*ch*ch+ch//32*4096+ch//32+ch
        sizes.update({name:655360 for name in BRIDGE_HASHES})
    return sizes


def _validate_tables(path, entries, expected):
    actual = {p.name for p in path.iterdir() if p.suffix in ('.f32','.i32')}
    if set(entries) != set(expected) or actual != set(expected):
        raise RuntimeError('incomplete or unexpected table set')
    for name,count in expected.items():
        entry = entries[name]
        file = path/name
        if file.is_symlink() or entry.get('count') != count or file.stat().st_size != count*4:
            raise RuntimeError(f'invalid table length: {name}')
        if sha256_file(file) != entry.get('sha256'):
            raise RuntimeError(f'table digest mismatch: {name}')
        if file.suffix == '.i32':
            values = [x[0] for x in struct.iter_unpack('<i',file.read_bytes())]
            if name not in BRIDGE_HASHES or entry['sha256'] != BRIDGE_HASHES[name] or sorted(values) != list(range(count)):
                raise RuntimeError(f'invalid derived bridge map: {name}')
        elif not all(math.isfinite(x[0]) for x in struct.iter_unpack('<f',file.read_bytes())):
            raise RuntimeError(f'nonfinite table: {name}')


def _table_provenance(name):
    return ('derived-bridge-not-original-640-captured' if name.endswith('.i32') else
            'confirmed-audit-contract-unchanged' if name in expected_tables() else
            'amd-consumer-derived-not-upstream-captured')


def validate_cache(path, *, audit_only=False, layout_mode=None):
    derived = _experimental(layout_mode)
    if audit_only and derived:
        raise ValueError('audit_only and experimental layout_mode are mutually exclusive')
    try:
        manifest = json.loads((Path(path)/'manifest.json').read_text())
        if derived:
            if (manifest.get('schema') != DERIVED_VERSION or manifest.get('source_sha256') != KNOWN_NVIDIA_SHA
                    or manifest.get('upstream_commit') != UPSTREAM_COMMIT or manifest.get('archive_records') != 153
                    or manifest.get('layout_mode') != layout_mode or manifest.get('runtime_ready') is not True
                    or manifest.get('experimental_runtime_ready') is not True or manifest.get('nvidia_equivalence') is not False
                    or manifest.get('provenance') != DERIVED_PROVENANCE or manifest.get('blockers') != []):
                raise RuntimeError('invalid experimental cache provenance or manifest')
            scalars = manifest.get('scalar_records',{})
            if (set(scalars) != {name for name,size in expected_records().items() if size == 2}
                    or not all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in scalars.values())):
                raise RuntimeError('invalid experimental scalar records')
            if any(entry.get('provenance') != _table_provenance(name) for name,entry in manifest['tables'].items()):
                raise RuntimeError('invalid experimental per-table provenance')
            _validate_tables(Path(path),manifest['tables'],expected_tables(layout_mode=layout_mode))
            if (Path(path)/'full-network.ok').read_text() != DERIVED_VERSION+'\n':
                raise RuntimeError('invalid experimental runtime marker')
            maps = [[x[0] for x in struct.iter_unpack('<i',(Path(path)/name).read_bytes())] for name in BRIDGE_HASHES]
            if any(maps[1][v] != i or maps[0][maps[1][i]] != i for i,v in enumerate(maps[0])):
                raise RuntimeError('derived bridge maps are not inverse')
            return manifest
        if not audit_only:
            # Runtime acceptance always requires the explicit experimental mode.
            raise RuntimeError('incomplete runtime cache: unresolved measured maps and ViT bridge')
        if (manifest['schema'] != CACHE_VERSION or manifest['source_sha256'] != KNOWN_NVIDIA_SHA
                or manifest['upstream_commit'] != UPSTREAM_COMMIT or manifest['runtime_ready'] is not False
                or manifest['archive_records'] != 153 or manifest['blockers'] != BLOCKERS):
            raise RuntimeError('invalid or stale cache manifest')
        _validate_tables(Path(path),manifest['tables'],expected_tables())
        if (Path(path)/'full-network.ok').exists():
            raise RuntimeError('incomplete audit has a runtime marker')
        return manifest
    except (OSError,KeyError,TypeError,ValueError) as exc:
        raise RuntimeError(f'invalid or incomplete cache: {exc}') from exc


def convert_nvidia_dll(dll_path: Path, output: Path | None = None, *, audit_only=False,
                       layout_mode=None, progress=None) -> Path:
    """Default fails closed. audit_only=True writes explicitly incomplete tables.

    layout_mode='amd-consumer-derived' enables an explicitly EXPERIMENTAL
    complete cache, not NVIDIA-equivalent or upstream-captured layouts.
    Never reuses the old unversioned cache. Existing output is verified, never
    repaired in place; stale/partial directories require a fresh output path.
    progress, if given, is called as progress(done, total, table_name).
    """
    derived = _experimental(layout_mode)
    if audit_only and derived:
        raise ValueError('audit_only and experimental layout_mode are mutually exclusive')
    source = Path(dll_path).expanduser().resolve()
    if not source.is_file():
        raise RuntimeError(f'NVIDIA DLL not found: {source}')
    if sha256_file(source) != KNOWN_NVIDIA_SHA:
        raise RuntimeError('unrecognized nvngx_dlssnr.dll; pinned SHA256 required')
    if not audit_only and not derived:
        raise RuntimeError('unresolved measured C32/multi attention maps and ViT bridge; audit_only extraction is not runnable')
    if output is None:
        base = Path(os.environ.get('XDG_CACHE_HOME',Path.home()/'.cache'))
        output = base/'dlss5-hip'/'weights'/(DERIVED_VERSION if derived else CACHE_VERSION)/KNOWN_NVIDIA_SHA
    cache = Path(output)
    if cache.exists():
        validate_cache(cache,audit_only=audit_only,layout_mode=layout_mode)
        return cache
    data = source.read_bytes()
    if len(data) < ARCHIVE_OFFSET+8:
        raise RuntimeError('DLL too small for WEIGHTS archive')
    size = _u64(data,ARCHIVE_OFFSET)
    if size < 8 or ARCHIVE_OFFSET+size > len(data):
        raise RuntimeError('WEIGHTS archive outside DLL bounds')
    archive = data[ARCHIVE_OFFSET:ARCHIVE_OFFSET+size]
    records = parse_archive(archive)
    payloads = {r['name']:archive[r['payload_offset']:r['payload_offset']+r['payload_size']] for r in records}
    # Validate scalar-only records even though native table contracts do not
    # consume ViT layer3 placeholders or post70 optional blend_scale here.
    scalars = {name:_f16s(raw)[0] for name,raw in payloads.items() if len(raw)==2}
    if not all(math.isfinite(v) for v in scalars.values()):
        raise RuntimeError('nonfinite scalar record')
    cache.mkdir(parents=True,exist_ok=False)
    expected,entries = expected_tables(layout_mode=layout_mode),{}
    total_tables = len(expected) + (len(BRIDGE_HASHES) if derived else 0)
    done_tables = 0
    def report(name):
        nonlocal done_tables
        done_tables += 1
        if progress is not None:
            progress(done_tables,total_tables,name)
    def payload(block,layer=0):
        return payloads[f'block{block}.layer{layer}.layer']
    def emit(name,values):
        if name in entries or name not in expected or len(values) != expected[name]:
            raise RuntimeError(f'wrong table contract: {name}')
        if not all(math.isfinite(v) for v in values):
            raise RuntimeError(f'nonfinite decoded coefficient: {name}')
        _write_f32(cache/name,values)
        entries[name] = {'count':len(values),'sha256':sha256_file(cache/name)}
        report(name)
    mix,pre_body = _preblock(payload(0))
    emit('block0-mix.audit.f32',mix)
    post_body,scales,head = _post70(payload(70))
    emit('post70-head.f32',head)
    emit('post70-scales.f32',scales)
    bodies = {0:pre_body,70:post_body}
    for b,ch in [(48,256),(56,128),(62,64),(66,32)]:
        weights,body = _upsample(payload(b),ch)
        emit(f'block{b}-weights.f32',weights)
        bodies[b] = body
    for b,ch in _multi_blocks():
        raw = bodies.get(b,payload(b))
        ffn,attention = _unpack_multi(raw,ch,layout_mode=layout_mode)
        emit(f'block{b}-ffn.f32',ffn)
        emit(f'block{b}-attention-components.audit.f32',_multi_attention_components(raw,ch))
        if derived:
            emit(f'block{b}-attention.f32',attention)
    if derived:
        for b in list(range(5))+list(range(66,71)):
            ffn,attention = _unpack_c32(bodies.get(b,payload(b)),layout_mode=layout_mode)
            if b == 0:
                ffn[:512] = mix
            prefix = 'post70' if b == 70 else f'block{b}'
            emit(prefix+'-ffn.f32',ffn)
            emit(prefix+'-attention.f32',attention)
        emit('block4-ds.f32',_downsample(payload(4),32,layout_mode=layout_mode))
    for b,ch in [(8,64),(14,128),(22,256)]:
        emit(f'block{b}-ds.f32',_downsample(payload(b),ch))
    for b in list(range(23,31))+list(range(40,48)):
        fw,fp,aw = _split([payload(b,i) for i in range(4)])
        for suffix,values in [('ffwd',fw),('ffwd-projection',fp),('attention',aw)]:
            emit(f'block{b}-{suffix}.f32',values)
    emit('head-matrix.f32',_head(payload(30,4)))
    for b in range(31,39):
        emit(f'block{b}-expand.f32',_vit_expand(payload(b)))
        emit(f'block{b}-contract.f32',_vit_residual(payload(b,1),4096))
        emit(f'block{b}-qkv.f32',_vit_qkv(payload(b,2)))
        emit(f'block{b}-projection.f32',_vit_residual(payload(b,4),1024))
    emit('decoder39-weights.f32',_decoder39(payload(39)))
    if derived:
        for name,values in zip(BRIDGE_HASHES,_derived_bridge(),strict=True):
            if values.itemsize != 4:
                raise RuntimeError('32-bit integer array required')
            if sys.byteorder != 'little':
                values.byteswap()
            (cache/name).write_bytes(values.tobytes())
            entries[name] = {'count':len(values),'sha256':sha256_file(cache/name)}
            report(name)
    _validate_tables(cache,entries,expected)
    manifest: dict = dict(schema=CACHE_VERSION,source_sha256=KNOWN_NVIDIA_SHA,
                    upstream_commit=UPSTREAM_COMMIT,archive_records=len(records),
                    archive_sha256=hashlib.sha256(archive).hexdigest(),runtime_ready=False,
                    blockers=BLOCKERS,scalar_records=scalars,tables=entries)
    if derived:
        manifest.update(schema=DERIVED_VERSION,layout_mode=layout_mode,runtime_ready=True,
                        experimental_runtime_ready=True,nvidia_equivalence=False,
                        provenance=DERIVED_PROVENANCE,blockers=[])
        for name,entry in entries.items():
            entry['provenance'] = _table_provenance(name)
        (cache/'full-network.ok').write_text(DERIVED_VERSION+'\n')
    (cache/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    validate_cache(cache,audit_only=audit_only,layout_mode=layout_mode)
    return cache
