"""C1 line (2026-10-29): pe_server_ext + op 'c1_contacts' (offline diagnosis only, privileged, never reaches a policy).
PhysX scene-query overlap of selected colliders against every other collider in the stage: which bodies touch / interpenetrate.
  args: {match: [substr, ...] (collider paths to test; case-insensitive), inflate_m: float (also test spheres of this radius at the
         collider's world AABB corners? no: only overlap_shape / overlap_mesh), list: bool (return all collider paths)}
  -> {hits: {query_path: [hit_collider_path, ...]}, n_colliders, errors, [colliders]}"""
import pe_server_ext as PX
S = PX.S
_d = S.Session.dispatch
_COLL = None


def _colliders():
    global _COLL
    if _COLL is None:
        import omni.usd
        from pxr import UsdPhysics
        st = omni.usd.get_context().get_stage(); _COLL = []
        for p in st.Traverse():
            if p.HasAPI(UsdPhysics.CollisionAPI): _COLL.append((str(p.GetPath()), p.GetTypeName()))
    return _COLL


def contacts(args):
    from omni.physx import get_physx_scene_query_interface
    from pxr import PhysicsSchemaTools, Sdf
    sq = get_physx_scene_query_interface(); col = _colliders(); match = [m.lower() for m in args.get('match', [])]
    excl = [m.lower() for m in args.get('exclude', [])]
    out = dict(hits={}, n_colliders=len(col), errors={})
    for path, typ in col:
        lp = path.lower()
        if not any(m in lp for m in match) or any(e in lp for e in excl): continue
        hits = []
        def rep(h):
            c = str(getattr(h, 'collision', '') or getattr(h, 'rigid_body', ''))
            if c and c != path: hits.append(c)
            return True
        lo, hi = PhysicsSchemaTools.encodeSdfPath(Sdf.Path(path))
        try:
            if hasattr(sq, 'overlap_shape'): sq.overlap_shape(lo, hi, rep, False)
            elif typ == 'Mesh': sq.overlap_mesh(lo, hi, rep, False)
            else: out['errors'][path] = 'no overlap_shape for %s' % typ; continue
        except Exception as e:  # noqa: BLE001
            out['errors'][path] = '%s: %s' % (type(e).__name__, str(e)[:160]); continue
        out['hits'][path] = sorted(set(hits))
    # positive control + coarse proximity: sphere overlaps at caller-given world points {name: [x, y, z, r]}
    import carb
    out['spheres'] = {}
    for name, (x, y, z, r) in (args.get('spheres') or {}).items():
        hs = []
        def rep2(h):
            c = str(getattr(h, 'collision', '') or getattr(h, 'rigid_body', ''))
            if c: hs.append(c)
            return True
        try:
            sq.overlap_sphere(float(r), carb.Float3(float(x), float(y), float(z)), rep2, False); out['spheres'][name] = sorted(set(hs))
        except Exception as e:  # noqa: BLE001
            out['errors']['sphere:' + name] = '%s: %s' % (type(e).__name__, str(e)[:160])
    if args.get('list'): out['colliders'] = col
    return out


def dispatch(self, op, args):
    if op == 'c1_contacts':
        return contacts(args)
    return _d(self, op, args)


S.Session.dispatch = dispatch
if __name__ == '__main__':
    S.main()
