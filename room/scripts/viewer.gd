extends Node3D
# Orbit viewer for the room.
# Drag = spin, wheel / trackpad scroll / pinch / Up-Down arrows / +- = zoom, right-drag = pan, C = toggle ceiling, R = reset, W = walk mode.

var room: Node
var cam: Camera3D
var target := Vector3(0, 1.2, -1.0)
var yaw := 0.3
var pitch := -0.45
var dist := 17.0


func _ready() -> void:
	room = load("res://scenes/room.tscn").instantiate()
	add_child(room)
	room.get_node("Player").process_mode = Node.PROCESS_MODE_DISABLED
	cam = Camera3D.new()
	cam.fov = 60.0
	cam.far = 200.0
	add_child(cam)
	cam.current = true
	room.get_node("Ceiling").visible = false
	_update()


func _update() -> void:
	var off := Vector3(0, 0, dist).rotated(Vector3.RIGHT, pitch).rotated(Vector3.UP, yaw)
	cam.global_position = target + off
	cam.look_at(target)


func _unhandled_input(e: InputEvent) -> void:
	if e is InputEventMouseMotion:
		if e.button_mask & MOUSE_BUTTON_MASK_LEFT:
			yaw -= e.relative.x * 0.006
			pitch = clampf(pitch - e.relative.y * 0.006, -1.5, 1.5)
		elif e.button_mask & MOUSE_BUTTON_MASK_RIGHT:
			var r := cam.global_transform.basis
			target += (-r.x * e.relative.x + r.y * e.relative.y) * dist * 0.0015
		_update()
	elif e is InputEventPanGesture:
		# Mac trackpad two-finger scroll
		dist = clampf(dist * (1.0 + e.delta.y * 0.05), 0.3, 40.0)
		_update()
	elif e is InputEventMagnifyGesture:
		# Mac trackpad pinch
		dist = clampf(dist / maxf(e.factor, 0.1), 0.3, 40.0)
		_update()
	elif e is InputEventMouseButton and e.pressed:
		if e.button_index == MOUSE_BUTTON_WHEEL_UP:
			dist = maxf(0.3, dist * 0.9)
		elif e.button_index == MOUSE_BUTTON_WHEEL_DOWN:
			dist = minf(40.0, dist * 1.1)
		_update()
	elif e is InputEventKey and e.pressed:
		match e.keycode:
			KEY_UP, KEY_EQUAL, KEY_KP_ADD:
				dist = maxf(0.3, dist * 0.85)
				_update()
			KEY_DOWN, KEY_MINUS, KEY_KP_SUBTRACT:
				dist = minf(40.0, dist * 1.15)
				_update()
			KEY_C:
				room.get_node("Ceiling").visible = not room.get_node("Ceiling").visible
			KEY_R:
				yaw = 0.3
				pitch = -0.45
				dist = 17.0
				target = Vector3(0, 1.2, -1.0)
				_update()
			KEY_W:
				get_tree().change_scene_to_file("res://scenes/room.tscn")
