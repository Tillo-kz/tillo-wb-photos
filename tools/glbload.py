"""Чтение GLB из Hunyuan: V (Hunyuan: верх = -Z после V=(x,-y,-z)), F, UV, текстура (BGR)."""
import json, struct
import numpy as np, cv2

_CT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_NC = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}


def load_glb(path):
    b = open(path, 'rb').read()
    assert b[:4] == b'glTF'
    off = 12; J = None; BIN = None
    while off < len(b):
        ln, typ = struct.unpack('<II', b[off:off + 8]); ch = b[off + 8:off + 8 + ln]
        if typ == 0x4E4F534A: J = json.loads(ch)
        elif typ == 0x004E4942: BIN = ch
        off += 8 + ln

    def acc(i):
        a = J['accessors'][i]; bv = J['bufferViews'][a['bufferView']]
        st = bv.get('byteOffset', 0) + a.get('byteOffset', 0)
        dt = _CT[a['componentType']]; nc = _NC[a['type']]
        stride = bv.get('byteStride')
        if stride and stride != np.dtype(dt).itemsize * nc:
            raw = np.frombuffer(BIN, np.uint8, count=stride * a['count'], offset=st).reshape(a['count'], stride)
            return raw[:, :np.dtype(dt).itemsize * nc].copy().view(dt).reshape(a['count'], nc)
        return np.frombuffer(BIN, dt, count=a['count'] * nc, offset=st).reshape(a['count'], nc)

    Vs, Fs, UVs, tex = [], [], [], None
    base = 0
    for m in J['meshes']:
        for p in m['primitives']:
            V = acc(p['attributes']['POSITION']).astype(float)
            F = acc(p['indices']).reshape(-1, 3).astype(np.int64) if 'indices' in p else np.arange(len(V)).reshape(-1, 3)
            UV = acc(p['attributes']['TEXCOORD_0']).astype(float) if 'TEXCOORD_0' in p['attributes'] else np.zeros((len(V), 2))
            Vs.append(V); Fs.append(F + base); UVs.append(UV); base += len(V)
            if tex is None and 'material' in p:
                mat = J['materials'][p['material']]
                ti = mat.get('pbrMetallicRoughness', {}).get('baseColorTexture', {}).get('index')
                if ti is not None:
                    img = J['images'][J['textures'][ti]['source']]
                    bv = J['bufferViews'][img['bufferView']]
                    data = BIN[bv.get('byteOffset', 0):bv.get('byteOffset', 0) + bv['byteLength']]
                    tex = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    V = np.vstack(Vs); F = np.vstack(Fs); UV = np.vstack(UVs)
    V = np.c_[V[:, 0], -V[:, 1], -V[:, 2]]
    return V, F, UV, tex, J


def weld(V, F, UV, tol=1e-4):
    q = np.round(V / tol).astype(np.int64)
    _, idx, inv = np.unique(q, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    F2 = inv[F]
    ok = (F2[:, 0] != F2[:, 1]) & (F2[:, 1] != F2[:, 2]) & (F2[:, 0] != F2[:, 2])
    return V[idx], F2[ok], ok
