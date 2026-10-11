extends SceneTree
# Generates res://scenes/room.tscn - the "Multi-Model Hackathon" stage room.
# Run: godot --headless --path . -s tools/build_room.gd
#
# Coordinates (metres): +X = east (right of the screen), -Z = north (screen wall), Y up.
# Room interior: x -7..7, z -6..6, ceiling 4.2.

const RX := 7.0
const RZ := 6.0
const H := 4.2
const WT := 0.3  # wall thickness

var scene_root: Node3D
var mats := {}
var manifest: Array = []  # exported to room/export/room_manifest.json


func _init() -> void:
	scene_root = Node3D.new()
	scene_root.name = "Room"
	_build_shell()
	_build_stage()
	_build_screen_wall_details()
	_build_ceiling()
	_build_audience()
	_build_lights()
	_build_player()
	var ps := PackedScene.new()
	var err := ps.pack(scene_root)
	if err != OK:
		push_error("pack failed %d" % err)
		quit(1)
		return
	err = ResourceSaver.save(ps, "res://scenes/room.tscn")
	_write_manifest()
	print("saved room.tscn err=", err)
	quit(0 if err == OK else 1)


# ---------------------------------------------------------------- manifest
# One entry per object the robot can see or bump into. x,z = centre, sx,sz = footprint.
func reg(id: String, kind: String, pos: Vector3, size: Vector3, solid := true, rot_y := 0.0) -> void:
	manifest.append({"id": id, "name": id.replace("_", " "), "kind": kind, "x": snappedf(pos.x, 0.001), "z": snappedf(pos.z, 0.001),
		"sx": snappedf(size.x, 0.001), "sz": snappedf(size.z, 0.001), "height": snappedf(size.y, 0.001), "solid": solid, "rot": rot_y})


func _write_manifest() -> void:
	DirAccess.make_dir_recursive_absolute("res://export")
	var data := {"bounds": {"minX": -RX, "maxX": RX, "minZ": -RZ, "maxZ": RZ}, "height": H, "objects": manifest}
	var f := FileAccess.open("res://export/room_manifest.json", FileAccess.WRITE)
	f.store_string(JSON.stringify(data, "  "))
	f.close()
	print("manifest objects=", manifest.size())


# ---------------------------------------------------------------- helpers
func mat(color: Color, tex: String = "", uv: Vector3 = Vector3.ONE, emit: float = 0.0, rough: float = 0.85) -> StandardMaterial3D:
	var key := "%s|%s|%s|%s|%s" % [color, tex, uv, emit, rough]
	if mats.has(key):
		return mats[key]
	var m := StandardMaterial3D.new()
	m.albedo_color = color
	m.roughness = rough
	if tex != "":
		m.albedo_texture = load(tex)
		m.uv1_scale = uv
		m.texture_filter = BaseMaterial3D.TEXTURE_FILTER_LINEAR_WITH_MIPMAPS_ANISOTROPIC
	if emit > 0.0:
		m.emission_enabled = true
		m.emission = color
		m.emission_energy_multiplier = emit
	mats[key] = m
	return m


# PBR material: tile_m = real-world size of one texture tile (box UVs are generated in tile units).
func pbr(set_name: String, tile_m: float, tint := Color(1, 1, 1), normal_scale := 1.0) -> StandardMaterial3D:
	var key := "pbr|%s|%s|%s" % [set_name, tile_m, tint]
	if mats.has(key):
		return mats[key]
	var m := StandardMaterial3D.new()
	m.albedo_texture = load("res://assets/tex/%s_albedo.png" % set_name)
	m.albedo_color = tint
	m.normal_enabled = true
	m.normal_texture = load("res://assets/tex/%s_normal.png" % set_name)
	m.normal_scale = normal_scale
	m.roughness = 1.0
	m.roughness_texture = load("res://assets/tex/%s_rough.png" % set_name)
	m.roughness_texture_channel = BaseMaterial3D.TEXTURE_CHANNEL_RED
	m.set_meta("tile", tile_m)  # box meshes get UVs in tile units (exportable to glTF)
	m.texture_filter = BaseMaterial3D.TEXTURE_FILTER_LINEAR_WITH_MIPMAPS_ANISOTROPIC
	mats[key] = m
	return m


# Instance a GLB asset (saved as an instance reference, not copied into the scene).
func inst(path: String, parent: Node, pos: Vector3, rot_y := 0.0, nm := "") -> Node3D:
	var n: Node3D = (load(path) as PackedScene).instantiate()
	if nm != "":
		n.name = nm
	n.position = pos
	n.rotation_degrees.y = rot_y
	own(n, parent)
	return n


func model(nm: String) -> String:
	return "res://assets/models/%s.glb" % nm


func own(n: Node, parent: Node) -> void:
	parent.add_child(n)
	n.owner = scene_root


func group(name: String, parent: Node = null, pos := Vector3.ZERO, rot_y := 0.0) -> Node3D:
	var g := Node3D.new()
	g.name = name
	g.position = pos
	g.rotation.y = deg_to_rad(rot_y)
	own(g, parent if parent else scene_root)
	return g



# Box mesh with UVs in real-world tile units when the material carries a "tile" meta.
func box_mesh(size: Vector3, m: Material) -> Mesh:
	if m == null or not m.has_meta("tile"):
		var bm := BoxMesh.new()
		bm.size = size
		return bm
	var t: float = m.get_meta("tile")
	var h := size * 0.5
	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)
	# [normal, u-axis, v-axis] per face, with u x v = normal (clockwise winding via the index order below)
	var faces := [
		[Vector3(0, 1, 0), Vector3(1, 0, 0), Vector3(0, 0, -1)],
		[Vector3(0, -1, 0), Vector3(1, 0, 0), Vector3(0, 0, 1)],
		[Vector3(1, 0, 0), Vector3(0, 0, -1), Vector3(0, 1, 0)],
		[Vector3(-1, 0, 0), Vector3(0, 0, 1), Vector3(0, 1, 0)],
		[Vector3(0, 0, 1), Vector3(1, 0, 0), Vector3(0, 1, 0)],
		[Vector3(0, 0, -1), Vector3(-1, 0, 0), Vector3(0, 1, 0)],
	]
	for f in faces:
		var n: Vector3 = f[0]
		var u: Vector3 = f[1]
		var v: Vector3 = f[2]
		var ue: float = absf(u.dot(size))
		var ve: float = absf(v.dot(size))
		var c := n * (absf(n.dot(size)) * 0.5)
		var corners := [c - u * ue * 0.5 - v * ve * 0.5, c + u * ue * 0.5 - v * ve * 0.5, c + u * ue * 0.5 + v * ve * 0.5, c - u * ue * 0.5 + v * ve * 0.5]
		var uvs := [Vector2(0, ve / t), Vector2(ue / t, ve / t), Vector2(ue / t, 0), Vector2(0, 0)]
		for idx in [0, 2, 1, 0, 3, 2]:
			st.set_normal(n)
			st.set_uv(uvs[idx])
			st.add_vertex(corners[idx])
	st.generate_tangents()
	return st.commit()

# Visual-only box (no collision).
func vbox(name: String, parent: Node, pos: Vector3, size: Vector3, m: Material, rot := Vector3.ZERO) -> MeshInstance3D:
	var mi := MeshInstance3D.new()
	mi.name = name
	mi.mesh = box_mesh(size, m)
	mi.material_override = m
	mi.position = pos
	mi.rotation_degrees = rot
	own(mi, parent)
	return mi


# Solid box: StaticBody3D + mesh + collision.
func sbox(name: String, parent: Node, pos: Vector3, size: Vector3, m: Material, rot := Vector3.ZERO) -> StaticBody3D:
	var sb := StaticBody3D.new()
	sb.name = name
	sb.position = pos
	sb.rotation_degrees = rot
	own(sb, parent)
	var mi := MeshInstance3D.new()
	mi.name = "Mesh"
	mi.mesh = box_mesh(size, m)
	mi.material_override = m
	own(mi, sb)
	var cs := CollisionShape3D.new()
	cs.name = "Collision"
	var bs := BoxShape3D.new()
	bs.size = size
	cs.shape = bs
	own(cs, sb)
	return sb


# Collision-only box (invisible blocker) - for chairs/props whose mesh is detailed.
func cbox(name: String, parent: Node, pos: Vector3, size: Vector3) -> StaticBody3D:
	var sb := StaticBody3D.new()
	sb.name = name
	sb.position = pos
	own(sb, parent)
	var cs := CollisionShape3D.new()
	cs.name = "Collision"
	var bs := BoxShape3D.new()
	bs.size = size
	cs.shape = bs
	own(cs, sb)
	return sb


func vcyl(name: String, parent: Node, pos: Vector3, r_top: float, r_bot: float, h: float, m: Material) -> MeshInstance3D:
	var mi := MeshInstance3D.new()
	mi.name = name
	var cm := CylinderMesh.new()
	cm.top_radius = r_top
	cm.bottom_radius = r_bot
	cm.height = h
	cm.radial_segments = 20
	mi.mesh = cm
	mi.material_override = m
	mi.position = pos
	own(mi, parent)
	return mi


func vsphere(name: String, parent: Node, pos: Vector3, r: float, m: Material, sy := 1.0) -> MeshInstance3D:
	var mi := MeshInstance3D.new()
	mi.name = name
	var sm := SphereMesh.new()
	sm.radius = r
	sm.height = 2.0 * r * sy
	sm.radial_segments = 20
	sm.rings = 10
	mi.mesh = sm
	mi.material_override = m
	mi.position = pos
	own(mi, parent)
	return mi


func scyl(name: String, parent: Node, pos: Vector3, radius: float, h: float, m: Material) -> StaticBody3D:
	var sb := StaticBody3D.new()
	sb.name = name
	sb.position = pos
	own(sb, parent)
	vcyl("Mesh", sb, Vector3.ZERO, radius, radius, h, m)
	var cs := CollisionShape3D.new()
	cs.name = "Collision"
	var s := CylinderShape3D.new()
	s.radius = radius
	s.height = h
	cs.shape = s
	own(cs, sb)
	return sb


# Wall along X (axis "x", fixed z=c) or along Z (axis "z", fixed x=c), spanning [a,b] on its axis.
# holes: Array of [start, end, y_bottom, y_top] along the axis. Fills each hole with a glass pane.
func wall(name: String, axis: String, c: float, a: float, b: float, holes: Array, m: Material, parent: Node, h := H) -> Node3D:
	var g := group(name, parent)
	var cur := a
	var sorted := holes.duplicate()
	sorted.sort_custom(func(p, q): return p[0] < q[0])
	var idx := 0
	for hole in sorted:
		_wall_piece(g, axis, c, cur, hole[0], 0.0, h, m, "Pier%d" % idx)
		_wall_piece(g, axis, c, hole[0], hole[1], 0.0, hole[2], m, "Sill%d" % idx)
		_wall_piece(g, axis, c, hole[0], hole[1], hole[3], h, m, "Lintel%d" % idx)
		_glass(g, axis, c, hole[0], hole[1], hole[2], hole[3], idx)
		cur = hole[1]
		idx += 1
	_wall_piece(g, axis, c, cur, b, 0.0, h, m, "Pier%d" % idx)
	return g


func _wall_piece(g: Node, axis: String, c: float, s0: float, s1: float, y0: float, y1: float, m: Material, nm: String) -> void:
	if s1 - s0 < 0.01 or y1 - y0 < 0.01:
		return
	var mid := (s0 + s1) * 0.5
	var len := s1 - s0
	var hh := y1 - y0
	if axis == "x":
		sbox(nm, g, Vector3(mid, y0 + hh * 0.5, c), Vector3(len, hh, WT), m)
	else:
		sbox(nm, g, Vector3(c, y0 + hh * 0.5, mid), Vector3(WT, hh, len), m)


func _glass(g: Node, axis: String, c: float, s0: float, s1: float, y0: float, y1: float, idx: int) -> void:
	var gm := mat(Color(0.75, 0.9, 1.0, 0.35), "", Vector3.ONE, 0.0, 0.1)
	gm.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	var mid := (s0 + s1) * 0.5
	var len := s1 - s0
	var hh := y1 - y0
	var sky := mat(Color(0.6, 0.9, 1.0), "", Vector3.ONE, 1.5, 1.0)
	var out := -0.45 if c < 0.0 else 0.45
	if axis == "x":
		sbox("Glass%d" % idx, g, Vector3(mid, y0 + hh * 0.5, c), Vector3(len, hh, 0.05), gm)
		vbox("Sky%d" % idx, g, Vector3(mid, y0 + hh * 0.5, c + out), Vector3(len + 0.4, hh + 0.4, 0.02), sky)
	else:
		sbox("Glass%d" % idx, g, Vector3(c, y0 + hh * 0.5, mid), Vector3(0.05, hh, len), gm)
		vbox("Sky%d" % idx, g, Vector3(c + out, y0 + hh * 0.5, mid), Vector3(0.02, hh + 0.4, len + 0.4), sky)


# ---------------------------------------------------------------- shell
func _build_shell() -> void:
	var shell := group("Shell")
	var wood := pbr("wood_floor", 1.4, Color(0.85, 0.8, 0.76))
	var paper := pbr("wallpaper", 0.8, Color(0.92, 0.9, 0.86))
	var brick := pbr("brick", 1.0)
	sbox("Floor", shell, Vector3(0, -0.25, 0), Vector3(RX * 2 + WT * 2, 0.5, RZ * 2 + WT * 2), wood)
	# Window holes: 2.8 m wide, sill 0.3 m, top 2.9 m (matches window_curtain.glb)
	wall("WallNorth", "x", -RZ - WT * 0.5, -RX - WT, RX + WT, [[4.1, 6.9, 0.3, 2.9]], paper, shell)
	wall("WallWest", "z", -RX - WT * 0.5, -RZ, RZ, [[-4.4, -1.6, 0.3, 2.9], [0.2, 3.0, 0.3, 2.9]], paper, shell)
	wall("WallEast", "z", RX + WT * 0.5, -RZ, RZ, [[-4.4, -1.6, 0.3, 2.9], [0.2, 3.0, 0.3, 2.9]], paper, shell)
	wall("WallSouth", "x", RZ + WT * 0.5, -RX - WT, RX + WT, [[3.5, 5.5, 0.0, 2.6]], brick, shell)
	# Window units (frame, cyan reveal, sill, black curtain)
	inst(model("window_curtain"), shell, Vector3(5.5, 0.3, -RZ), 0.0, "WindowNorth")
	inst(model("window_curtain"), shell, Vector3(-RX, 0.3, -3.0), 90.0, "WindowWest1")
	inst(model("window_curtain"), shell, Vector3(-RX, 0.3, 1.6), 90.0, "WindowWest2")
	inst(model("window_curtain"), shell, Vector3(RX, 0.3, -3.0), -90.0, "WindowEast1")
	inst(model("window_curtain"), shell, Vector3(RX, 0.3, 1.6), -90.0, "WindowEast2")
	# Brick wing top-left of the screen wall + baseboards
	vbox("BrickWingLeft", shell, Vector3(-5.2, 2.1, -RZ + 0.02), Vector3(3.6, 4.2, 0.04), brick)
	var dark := mat(Color(0.06, 0.05, 0.05), "", Vector3.ONE, 0.0, 0.5)
	vbox("BaseboardN", shell, Vector3(0, 0.07, -RZ + 0.03), Vector3(RX * 2, 0.14, 0.06), dark)
	vbox("BaseboardS", shell, Vector3(0, 0.07, RZ - 0.03), Vector3(RX * 2, 0.14, 0.06), dark)
	vbox("DoorFrame", shell, Vector3(4.5, 1.3, RZ - 0.02), Vector3(2.1, 2.7, 0.06), mat(Color(0.05, 0.3, 0.3)))


# ---------------------------------------------------------------- stage
const STAGE_POS := Vector3(0.4, 0.0, -3.7)

func _build_stage() -> void:
	var st := group("Stage", null, STAGE_POS)
	# Modular deck (skirt on the audience side = +Z, so the model is turned 180 deg)
	inst(model("stage_deck"), st, Vector3.ZERO, 180.0, "Deck")
	cbox("DeckCollision", st, Vector3(0, 0.175, 0), Vector3(6.4, 0.35, 3.8))
	reg("stage", "platform", STAGE_POS, Vector3(6.4, 0.35, 3.8), false)
	# Rug (real photo of the rug, cleaned) lying on the plywood top
	var rug := StandardMaterial3D.new()
	rug.albedo_texture = load("res://assets/tex/rug_albedo.jpg")
	rug.normal_enabled = true
	rug.normal_texture = load("res://assets/tex/rug_normal.png")
	rug.normal_scale = 0.2
	rug.roughness = 1.0
	vbox("Rug", st, Vector3(0.0, 0.362, 0.05), Vector3(5.4, 0.012, 2.6), rug)
	# Access ramp on the audience side
	sbox("Ramp", st, Vector3(1.0, 0.175, 2.9), Vector3(1.4, 0.06, 1.95), mat(Color(0.32, 0.26, 0.2), "", Vector3.ONE, 0.0, 0.7), Vector3(10.3, 0, 0))
	# Seating on the stage (facing the audience, slightly turned in)
	var y := 0.35
	inst(model("chair_navy"), st, Vector3(-1.9, y, 0.1), 205.0, "ChairLeft")
	cbox("ChairLeftCol", st, Vector3(-1.9, y + 0.43, 0.1), Vector3(0.55, 0.85, 0.55))
	reg("chair_navy_left", "chair", STAGE_POS + Vector3(-1.9, 0, 0.1), Vector3(0.55, 0.85, 0.55), true, 205.0)
	inst(model("chair_navy"), st, Vector3(-0.5, y, -0.1), 180.0, "ChairMiddle")
	cbox("ChairMiddleCol", st, Vector3(-0.5, y + 0.43, -0.1), Vector3(0.55, 0.85, 0.55))
	reg("chair_navy_middle", "chair", STAGE_POS + Vector3(-0.5, 0, -0.1), Vector3(0.55, 0.85, 0.55), true, 180.0)
	inst(model("armchair_blue"), st, Vector3(1.1, y, 0.0), 155.0, "Armchair")
	cbox("ArmchairCol", st, Vector3(1.1, y + 0.55, 0.0), Vector3(0.75, 1.1, 0.7))
	reg("armchair_blue", "chair", STAGE_POS + Vector3(1.1, 0, 0.0), Vector3(0.75, 1.1, 0.7), true, 155.0)
	# Plants flanking the stage
	inst(model("plant_tall"), scene_root, Vector3(-3.7, 0, -4.9), 20.0, "PlantTall")
	cbox("PlantTallCol", scene_root, Vector3(-3.7, 1.0, -4.9), Vector3(0.6, 2.0, 0.6))
	reg("plant_tall", "plant", Vector3(-3.7, 0, -4.9), Vector3(0.6, 2.0, 0.6))
	inst(model("plant_medium"), scene_root, Vector3(4.6, 0, -3.6), 70.0, "PlantMedium")
	cbox("PlantMediumCol", scene_root, Vector3(4.6, 0.85, -3.6), Vector3(0.6, 1.7, 0.6))
	reg("plant_medium", "plant", Vector3(4.6, 0, -3.6), Vector3(0.6, 1.7, 0.6))


# ---------------------------------------------------------------- north wall details
func _build_screen_wall_details() -> void:
	var g := group("ScreenWall")
	# Ceiling projector screen (origin = top centre of the roller case)
	var sx := 0.4
	var top_y := 3.55
	var sz := -RZ + 0.16
	inst(model("projector_screen"), g, Vector3(sx, top_y, sz), 180.0, "ProjectorScreen")
	# Slide on the screen surface (4.0 x 2.25 m, centre 1.425 m below the case top)
	var slide := StandardMaterial3D.new()
	slide.albedo_texture = load("res://assets/slide.jpg")
	slide.albedo_color = Color(0.2, 0.2, 0.2)
	slide.emission_enabled = true
	slide.emission_texture = load("res://assets/slide.jpg")
	slide.emission_energy_multiplier = 0.8
	slide.roughness = 0.6
	var q := MeshInstance3D.new()
	q.name = "Slide"
	var qm := QuadMesh.new()
	qm.size = Vector2(4.0, 2.25)
	q.mesh = qm
	q.material_override = slide
	q.position = Vector3(sx, top_y - 1.425, sz + 0.004)
	own(q, g)
	# Neon sign above the screen
	vbox("NeonSign", g, Vector3(sx, 3.85, -RZ + 0.05), Vector3(1.5, 0.1, 0.03), mat(Color(0.75, 0.3, 1.0), "", Vector3.ONE, 3.0))
	# Orange accent panel + pale blue column
	vbox("AccentPanel", g, Vector3(-2.7, 1.9, -RZ + 0.03), Vector3(1.0, 3.8, 0.05), mat(Color(0.72, 0.25, 0.14), "", Vector3.ONE, 0.0, 0.8))
	sbox("BlueColumn", g, Vector3(3.85, H * 0.5, -RZ + 0.3), Vector3(0.45, H, 0.6), mat(Color(0.72, 0.85, 0.9), "", Vector3.ONE, 0.0, 0.7))
	reg("blue_column", "column", Vector3(3.85, 0, -RZ + 0.3), Vector3(0.45, H, 0.6))


# ---------------------------------------------------------------- ceiling
func _build_ceiling() -> void:
	var g := group("Ceiling")
	sbox("Slab", g, Vector3(0, H + 0.25, 0), Vector3(RX * 2 + WT * 2, 0.5, RZ * 2 + WT * 2), pbr("concrete", 2.0))
	# Concrete beams along Z (4 m each, 3 in a row), at 4 x positions
	for bx in [-5.25, -1.75, 1.75, 5.25]:
		for bz in [-4.0, 0.0, 4.0]:
			var b := inst(model("exposed_beam"), g, Vector3(bx, H - 0.3, bz), 90.0, "Beam_%s_%s" % [bx, bz])
	# Cable tray along Z, round duct along X
	for bz in [-4.0, 0.0, 4.0]:
		inst(model("cable_tray"), g, Vector3(-0.2, H - 0.68, bz), 90.0, "Tray_%s" % bz)
	for dx in [-5.25, -1.75, 1.75, 5.25]:
		inst(model("duct_round"), g, Vector3(dx, H - 0.95, 2.6), 0.0, "Duct_%s" % dx)
	# Hanging PA speakers (grille toward +Z = audience) and stage spot cans
	inst(model("pa_speaker_hanging"), g, Vector3(-3.5, H - 0.6, -2.4), 0.0, "PA_Left")
	inst(model("pa_speaker_hanging"), g, Vector3(4.3, H - 0.6, -2.4), 0.0, "PA_Right")
	for p in [Vector3(-5.25, H - 0.6, -2.6), Vector3(-1.75, H - 0.6, -2.6), Vector3(1.75, H - 0.6, -2.6), Vector3(5.25, H - 0.6, -2.6)]:
		inst(model("spot_can"), g, p, 180.0, "Spot_%s" % p.x)
	# Pendant dome lamps either side of the screen
	for x in [-1.75, 2.55]:
		inst(model("pendant_dome"), g, Vector3(x, H - 0.6, -4.7), 0.0, "Pendant_%s" % x)


# ---------------------------------------------------------------- audience
func prop(nm: String, model_name: String, parent: Node, pos: Vector3, rot_y: float, col_size: Vector3) -> void:
	inst(model(model_name), parent, pos, rot_y, nm)
	reg(nm.to_snake_case(), model_name.split("_")[0], pos, col_size, true, rot_y)
	cbox(nm + "Col", parent, pos + Vector3(0, col_size.y * 0.5, 0), col_size)


func _build_audience() -> void:
	var g := group("Audience")
	# 4 rows x 7 grey folding chairs facing the stage (-Z)
	for r in range(4):
		for c in range(7):
			var x := -5.7 + c * 0.7 + (0.0 if r % 2 == 0 else 0.35)
			var z := 0.4 + r * 1.05
			prop("FoldChair_%d_%d" % [r, c], "folding_chair_grey", g, Vector3(x, 0, z), 0.0, Vector3(0.5, 0.85, 0.55))
	# Table with white chairs behind it; the coffee cup (added by the web viewer) sits on it
	prop("Table", "table_folding", g, Vector3(4.6, 0, 1.4), 0.0, Vector3(1.8, 0.74, 0.7))
	reg("coffee_cup", "item", Vector3(4.3, 0.74, 1.4), Vector3(0.1, 0.1, 0.1), false)
	for i in range(3):
		prop("TableChair%d" % i, "folding_chair_white", g, Vector3(4.0 + i * 0.6, 0, 0.6), 180.0, Vector3(0.5, 0.85, 0.55))
	# Speaker on stand, camera tripod, lectern
	prop("SpeakerStand", "speaker_stand", g, Vector3(-1.2, 0, -0.4), 160.0, Vector3(0.6, 1.8, 0.6))
	prop("Tripod", "tripod_camera", g, Vector3(-3.0, 0, -0.9), 0.0, Vector3(0.6, 1.5, 0.6))
	prop("Lectern", "lectern_white", g, Vector3(3.0, 0, -0.4), 180.0, Vector3(0.7, 1.1, 0.5))


# ---------------------------------------------------------------- lights
func _build_lights() -> void:
	var g := group("Lights")
	var env := WorldEnvironment.new()
	env.name = "WorldEnvironment"
	var e := Environment.new()
	e.background_mode = Environment.BG_COLOR
	e.background_color = Color(0.02, 0.01, 0.03)
	e.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	e.ambient_light_color = Color(0.5, 0.42, 0.55)
	e.ambient_light_energy = 0.35
	e.tonemap_mode = Environment.TONE_MAPPER_ACES
	e.tonemap_exposure = 1.0
	e.glow_enabled = true
	e.glow_intensity = 0.5
	e.glow_bloom = 0.05
	e.ssao_enabled = true
	e.ssao_radius = 1.2
	e.ssao_intensity = 2.0
	e.ssil_enabled = true
	e.volumetric_fog_enabled = true
	e.volumetric_fog_density = 0.004
	e.volumetric_fog_albedo = Color(0.8, 0.7, 0.9)
	env.environment = e
	own(env, g)
	var rp := ReflectionProbe.new()
	rp.name = "ReflectionProbe"
	rp.position = Vector3(0, 2.0, 0)
	rp.size = Vector3(RX * 2, H, RZ * 2)
	rp.interior = true
	rp.box_projection = true
	rp.update_mode = ReflectionProbe.UPDATE_ONCE
	own(rp, g)
	# Coloured stage wash: spot cans aimed at the stage, with shadows
	var cols := [Color(1.0, 0.2, 0.85), Color(0.6, 0.3, 1.0), Color(0.3, 0.8, 1.0), Color(1.0, 0.35, 0.6)]
	var i := 0
	for x in [-5.25, -1.75, 1.75, 5.25]:
		var sl := SpotLight3D.new()
		sl.name = "StageSpot%d" % i
		sl.position = Vector3(x, H - 0.8, -2.8)
		sl.light_color = cols[i]
		sl.light_energy = 6.0
		sl.spot_range = 8.0
		sl.spot_angle = 38.0
		sl.shadow_enabled = true
		own(sl, g)
		sl.look_at_from_position(sl.position, Vector3(0.4 + (x - 0.0) * -0.15, 0.4, -4.4))
		i += 1
	# Window daylight (cyan), pendants (warm), general fill
	for w in [["WinN", Vector3(5.5, 2.0, -5.5)], ["WinW1", Vector3(-6.4, 2.0, -3.0)], ["WinW2", Vector3(-6.4, 2.0, 1.6)], ["WinE1", Vector3(6.4, 2.0, -3.0)], ["WinE2", Vector3(6.4, 2.0, 1.6)]]:
		_omni(w[0], g, w[1], Color(0.55, 0.9, 1.0), 3.0, 7.0)
	for x in [-1.75, 2.55]:
		_omni("PendantLight_%s" % x, g, Vector3(x, H - 0.6 - 1.3, -4.7), Color(1.0, 0.75, 0.4), 2.5, 5.0)
	_omni("RoomFill", g, Vector3(0, 3.4, 2.5), Color(1.0, 0.9, 0.8), 1.2, 12.0)


func _omni(name: String, parent: Node, pos: Vector3, col: Color, energy: float, rng: float) -> void:
	var l := OmniLight3D.new()
	l.name = name
	l.position = pos
	l.light_color = col
	l.light_energy = energy
	l.omni_range = rng
	own(l, parent)


# ---------------------------------------------------------------- player
func _build_player() -> void:
	var pl := CharacterBody3D.new()
	pl.name = "Player"
	pl.set_script(load("res://scripts/player.gd"))
	pl.position = Vector3(0.5, 0.1, 5.0)
	pl.collision_layer = 2
	own(pl, scene_root)
	var cs := CollisionShape3D.new()
	cs.name = "Collision"
	var cap := CapsuleShape3D.new()
	cap.radius = 0.28
	cap.height = 1.7
	cs.shape = cap
	cs.position = Vector3(0, 0.85, 0)
	own(cs, pl)
	var head := Node3D.new()
	head.name = "Head"
	head.position = Vector3(0, 1.6, 0)
	own(head, pl)
	var cam := Camera3D.new()
	cam.name = "Camera"
	cam.fov = 75.0
	cam.current = true
	own(cam, head)
