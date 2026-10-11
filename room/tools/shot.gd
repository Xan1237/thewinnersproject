extends SceneTree
# Renders screenshots: godot --path . -s tools/shot.gd  (needs a window, not headless)
const VIEWS := {
	"photo": [Vector3(0.4, 1.5, 3.2), Vector3(0.4, 1.7, -4.5), 80.0],
	"left": [Vector3(-5.5, 1.6, 4.5), Vector3(0.5, 1.2, -3.5), 70.0],
	"back": [Vector3(3.5, 1.7, -0.5), Vector3(-3.0, 1.5, 2.5), 80.0],
	"stage": [Vector3(0.6, 1.3, -1.2), Vector3(0.4, 0.9, -4.2), 55.0],
	"top": [Vector3(0.0, 3.9, 5.8), Vector3(0.0, 0.0, -1.0), 70.0],
}

func _init() -> void:
	var scene: Node = load("res://scenes/room.tscn").instantiate()
	get_root().add_child(scene)
	scene.get_node("Player").queue_free()
	var cam := Camera3D.new()
	get_root().add_child(cam)
	cam.current = true
	DirAccess.make_dir_recursive_absolute("res://evidence")
	for k in VIEWS:
		var v = VIEWS[k]
		cam.position = v[0]
		cam.look_at_from_position(v[0], v[1])
		cam.fov = v[2]
		for i in 4:
			await process_frame
		var img := get_root().get_texture().get_image()
		img.save_png("res://evidence/%s.png" % k)
	quit()
