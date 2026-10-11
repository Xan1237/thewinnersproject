extends SceneTree
# Exports the room as a lights-free GLB for the web arena viewer.
# Run: godot --headless --path . -s tools/export_web.gd   (writes export/room.glb)

func _init() -> void:
	var scene: Node = load("res://scenes/room.tscn").instantiate()
	get_root().add_child(scene)
	var drop: Array = []
	_collect(scene, drop)
	for n in drop:
		n.get_parent().remove_child(n)
		n.free()
	DirAccess.make_dir_recursive_absolute("res://export")
	var doc := GLTFDocument.new()
	var st := GLTFState.new()
	var err := doc.append_from_scene(scene, st)
	if err == OK:
		err = doc.write_to_filesystem(st, "res://export/room.glb")
	print("export err=", err)
	quit(0 if err == OK else 1)


func _collect(n: Node, out: Array) -> void:
	if n is Light3D or n is WorldEnvironment or n is ReflectionProbe or n is CharacterBody3D or n is Camera3D or n is Marker3D:
		out.append(n)
		return
	for c in n.get_children():
		_collect(c, out)
