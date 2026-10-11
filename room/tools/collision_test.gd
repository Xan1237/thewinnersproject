extends SceneTree
# Headless collision test: godot --headless --path . -s tools/collision_test.gd
var fails := 0

func check(label: String, ok: bool, info: String) -> void:
	print(("PASS " if ok else "FAIL "), label, " ", info)
	if not ok: fails += 1

func run_walk(start: Vector3, dir: Vector2, yaw_deg: float, frames: int) -> Vector3:
	var scene: Node = load("res://scenes/room.tscn").instantiate()
	get_root().add_child(scene)
	var pl: CharacterBody3D = scene.get_node("Player")
	pl.global_position = start
	pl.rotation.y = deg_to_rad(yaw_deg)
	pl.test_dir = dir
	for i in frames:
		await physics_frame
	var p := pl.global_position
	scene.queue_free()
	await process_frame
	return p

func _init() -> void:
	var p: Vector3
	p = await run_walk(Vector3(0.5, 0.05, 5.0), Vector2(0, -1), 0, 600)
	check("stage front blocked at x=0.5", p.z > -1.8 and p.y < 0.2, str(p))
	p = await run_walk(Vector3(1.4, 0.05, 1.0), Vector2(0, -1), 0, 400)
	check("ramp climbs onto stage", p.y > 0.3 and p.z < -2.0, str(p))
	p = await run_walk(Vector3(-2.0, 0.05, 5.0), Vector2(0, -1), 0, 600)
	check("walk north left side stops (chairs/stage/wall)", p.z > -6.0 and p.y < 0.5, str(p))
	p = await run_walk(Vector3(0.0, 0.05, 5.0), Vector2(-1, 0), 0, 900)
	check("west wall stops", p.x > -7.0 and p.x < -6.0, str(p))
	p = await run_walk(Vector3(0.0, 0.05, 5.0), Vector2(1, 0), 0, 900)
	check("east wall stops", p.x < 7.0 and p.x > 5.0, str(p))
	p = await run_walk(Vector3(0.0, 0.05, 0.0), Vector2(0, 1), 0, 900)
	check("south wall stops", p.z < 6.0 and p.z > 4.5, str(p))
	p = await run_walk(Vector3(1.0, 0.05, 3.0), Vector2(0, 0), 0, 120)
	check("stands on floor (no fall)", absf(p.y) < 0.05, str(p))
	p = await run_walk(Vector3(-3.0, 0.05, -0.1), Vector2(0, -1), 0, 300)
	check("tripod blocks (x=-3.0 walking north stops south of z=-0.5)", p.z > -0.5 and p.z < -0.1, str(p))
	p = await run_walk(Vector3(3.0, 0.05, 3.0), Vector2(0, -1), 0, 400)
	check("lectern blocks (x=3.0)", p.z > -0.2, str(p))
	print("FAILS=", fails)
	quit(1 if fails else 0)
