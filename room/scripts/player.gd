extends CharacterBody3D
# First-person walker. WASD move, mouse look, Shift run, Esc releases the mouse.

const SPEED := 2.8
const RUN := 5.0
const GRAVITY := 18.0
const SENS := 0.0025

@onready var head: Node3D = $Head

# Headless tests can drive these instead of the keyboard.
var test_dir := Vector2.ZERO


func _ready() -> void:
	floor_snap_length = 0.35
	if DisplayServer.get_name() != "headless":
		Input.mouse_mode = Input.MOUSE_MODE_CAPTURED


func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventMouseMotion and Input.mouse_mode == Input.MOUSE_MODE_CAPTURED:
		rotate_y(-event.relative.x * SENS)
		head.rotation.x = clampf(head.rotation.x - event.relative.y * SENS, -1.4, 1.4)
	elif event is InputEventKey and event.pressed and event.keycode == KEY_ESCAPE:
		Input.mouse_mode = Input.MOUSE_MODE_VISIBLE if Input.mouse_mode == Input.MOUSE_MODE_CAPTURED else Input.MOUSE_MODE_CAPTURED
	elif event is InputEventMouseButton and event.pressed:
		Input.mouse_mode = Input.MOUSE_MODE_CAPTURED


func _physics_process(delta: float) -> void:
	var d := test_dir
	if d == Vector2.ZERO:
		d = Vector2(
			float(Input.is_physical_key_pressed(KEY_D)) - float(Input.is_physical_key_pressed(KEY_A)),
			float(Input.is_physical_key_pressed(KEY_S)) - float(Input.is_physical_key_pressed(KEY_W)))
	var spd := RUN if Input.is_physical_key_pressed(KEY_SHIFT) else SPEED
	var wish := (transform.basis * Vector3(d.x, 0, d.y)).normalized() * spd
	velocity.x = wish.x
	velocity.z = wish.z
	if not is_on_floor():
		velocity.y -= GRAVITY * delta
	else:
		velocity.y = 0.0
	move_and_slide()
