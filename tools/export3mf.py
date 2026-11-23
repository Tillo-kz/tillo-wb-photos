"""Экспорт многоцветной фигурки Tillo в 3MF-проект Bambu Studio с полной пластиной.

export_plate(V, F, Lf, colours, out_path, template_dir, name)
  V  (n,3) мм, низ в z=0; F (m,3); Lf (m,) экструдер 1..K на грань
  colours  ['#RRGGBB', ...] по экструдерам
  template_dir  распакованный проект владельца (Metadata/project_settings.config с тем же числом цветов)
Все экземпляры ссылаются на ОДНУ сетку (как у владельца) — проект лёгкий.
"""
import os, re, json, uuid, zipfile, shutil
import numpy as np

PAINT = {1: None, 2: "8", 3: "0C", 4: "1C"}  # экструдер 1 = цвет объекта по умолчанию

BED = 256.0
EDGE = 2.0
EXCL = (0, 0, 18 + 2, 28 + 2)               # bed_exclude_area + запас


def tower_zone(wx, wy, width=35.0, depth=45.0, brim=3.0, gap=5.0):
    return (wx - brim - gap, wy - brim - gap, wx + width + brim + gap, wy + depth + brim + gap)


def _hit(r, a):
    return not (r[2] <= a[0] or r[0] >= a[2] or r[3] <= a[1] or r[1] >= a[3])


def layout(sx, sy, gap=1.0, wipe=(206.0, 15.0)):
    """Сетка центров экземпляров. sx, sy — габарит фигурки (мм)."""
    tz = tower_zone(*wipe)
    best = []
    for rot in (0, 90):
        w, h = (sx, sy) if rot == 0 else (sy, sx)
        pts = []
        y = EDGE
        while y + h <= BED - EDGE + 1e-6:
            x = EDGE
            while x + w <= BED - EDGE + 1e-6:
                r = (x, y, x + w, y + h)
                if not _hit(r, EXCL) and not _hit(r, tz):
                    pts.append((x + w / 2, y + h / 2, rot))
                x += w + gap
            y += h + gap
        if len(pts) > len(best):
            best = pts
    return best


def _mesh_xml(V, F, Lf):
    out = ['<?xml version="1.0" encoding="UTF-8"?>\n<model unit="millimeter" xml:lang="en-US" '
           'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" '
           'xmlns:BambuStudio="http://schemas.bambulab.com/package/2021" '
           'xmlns:p="http://schemas.microsoft.com/3dmanufacturing/production/2015/06" requiredextensions="p">\n'
           ' <metadata name="BambuStudio:3mfVersion">1</metadata>\n <resources>\n'
           f'  <object id="1" p:UUID="{uuid.uuid4()}" type="model">\n   <mesh>\n    <vertices>\n']
    out += [f'     <vertex x="{v[0]:.5f}" y="{v[1]:.5f}" z="{v[2]:.5f}"/>\n' for v in V]
    out.append('    </vertices>\n    <triangles>\n')
    for f, l in zip(F, Lf):
        pc = PAINT[int(l)]
        a = f' paint_color="{pc}"' if pc else ''
        out.append(f'     <triangle v1="{f[0]}" v2="{f[1]}" v3="{f[2]}"{a}/>\n')
    out.append('    </triangles>\n   </mesh>\n  </object>\n </resources>\n <build/>\n</model>\n')
    return ''.join(out)


def export_plate(V, F, Lf, colours, out_path, template_dir, name, gap=1.0, wipe=(206.0, 15.0), max_n=None):
    V = np.asarray(V, float).copy()
    V[:, 0] -= (V[:, 0].min() + V[:, 0].max()) / 2
    V[:, 1] -= (V[:, 1].min() + V[:, 1].max()) / 2
    V[:, 2] -= V[:, 2].min()
    sx, sy = np.ptp(V[:, 0]), np.ptp(V[:, 1])
    pts = layout(sx, sy, gap, wipe)
    if max_n:
        pts = pts[:max_n]
    n = len(pts)

    tmp = out_path + '.d'
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp + '/3D/Objects'); os.makedirs(tmp + '/3D/_rels'); os.makedirs(tmp + '/_rels'); os.makedirs(tmp + '/Metadata')
    shutil.copy(os.path.join(template_dir, '[Content_Types].xml'), tmp)
    open(tmp + '/_rels/.rels', 'w').write(
        '<?xml version="1.0" encoding="UTF-8"?>\n<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
        ' <Relationship Target="/3D/3dmodel.model" Id="rel-1" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>\n</Relationships>')
    open(tmp + '/3D/_rels/3dmodel.model.rels', 'w').write(
        '<?xml version="1.0" encoding="UTF-8"?>\n<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
        ' <Relationship Target="/3D/Objects/object_1.model" Id="rel-1" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>\n</Relationships>')
    open(tmp + '/3D/Objects/object_1.model', 'w').write(_mesh_xml(V, F, Lf))

    zc = 0.0
    res, items, ms, inst, asm = [], [], [], [], []
    for i, (cx, cy, rot) in enumerate(pts):
        oid = i + 2
        c, s = (1, 0) if rot == 0 else (0, 1)
        tr = f"{c} {s} 0 {-s} {c} 0 0 0 1 {cx:.4f} {cy:.4f} {zc}"
        res.append(f'  <object id="{oid}" p:UUID="{uuid.uuid4()}" type="model">\n   <components>\n'
                   f'    <component p:path="/3D/Objects/object_1.model" objectid="1" p:UUID="{uuid.uuid4()}" transform="1 0 0 0 1 0 0 0 1 0 0 0"/>\n'
                   '   </components>\n  </object>\n')
        items.append(f'  <item objectid="{oid}" p:UUID="{uuid.uuid4()}" transform="{tr}" printable="1"/>\n')
        ms.append(f'  <object id="{oid}">\n    <metadata key="name" value="{name}_{i+1}"/>\n    <metadata key="extruder" value="1"/>\n'
                  f'    <metadata face_count="{len(F)}"/>\n    <part id="1" subtype="normal_part">\n      <metadata key="name" value="{name}"/>\n'
                  '      <metadata key="matrix" value="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"/>\n'
                  f'      <mesh_stat face_count="{len(F)}" edges_fixed="0" degenerate_facets="0" facets_removed="0" facets_reversed="0" backwards_edges="0"/>\n'
                  '    </part>\n  </object>\n')
        inst.append(f'    <model_instance>\n      <metadata key="object_id" value="{oid}"/>\n      <metadata key="instance_id" value="0"/>\n'
                    f'      <metadata key="identify_id" value="{1000+i}"/>\n    </model_instance>\n')
        asm.append(f'   <assemble_item object_id="{oid}" instance_id="0" transform="{tr}" offset="0 0 0" />\n')

    open(tmp + '/3D/3dmodel.model', 'w').write(
        '<?xml version="1.0" encoding="UTF-8"?>\n<model unit="millimeter" xml:lang="en-US" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" '
        'xmlns:BambuStudio="http://schemas.bambulab.com/package/2021" xmlns:p="http://schemas.microsoft.com/3dmanufacturing/production/2015/06" requiredextensions="p">\n'
        ' <metadata name="Application">BambuStudio-02.05.00.66</metadata>\n <metadata name="BambuStudio:3mfVersion">1</metadata>\n'
        f' <metadata name="Title">{name}</metadata>\n <resources>\n' + ''.join(res) + ' </resources>\n <build>\n' + ''.join(items) + ' </build>\n</model>\n')

    K = len(colours)
    open(tmp + '/Metadata/model_settings.config', 'w').write(
        '<?xml version="1.0" encoding="UTF-8"?>\n<config>\n' + ''.join(ms) +
        '  <plate>\n    <metadata key="plater_id" value="1"/>\n    <metadata key="plater_name" value=""/>\n    <metadata key="locked" value="false"/>\n'
        '    <metadata key="filament_map_mode" value="Auto For Flush"/>\n'
        f'    <metadata key="filament_maps" value="{" ".join(["1"]*K)}"/>\n    <metadata key="filament_volume_maps" value="{" ".join(["0"]*K)}"/>\n'
        + ''.join(inst) + '  </plate>\n  <assemble>\n' + ''.join(asm) + '  </assemble>\n</config>\n')

    cfg = json.load(open(os.path.join(template_dir, 'Metadata/project_settings.config')))
    assert len(cfg['filament_colour']) == K, 'шаблон с другим числом цветов'
    cfg['filament_colour'] = list(colours)
    cfg['wipe_tower_x'] = [f"{wipe[0]:g}"]
    cfg['wipe_tower_y'] = [f"{wipe[1]:g}"]
    cfg['enable_support'] = '0'
    json.dump(cfg, open(tmp + '/Metadata/project_settings.config', 'w'), indent=4, ensure_ascii=False)

    if os.path.exists(out_path):
        os.remove(out_path)
    with zipfile.ZipFile(out_path, 'w', zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(tmp):
            for fn in files:
                p = os.path.join(root, fn)
                z.write(p, os.path.relpath(p, tmp))
    shutil.rmtree(tmp)
    return n, (sx, sy)
