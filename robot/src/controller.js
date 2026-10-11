import * as THREE from 'three';

const UP = new THREE.Vector3(0, 1, 0);
const MOVEMENT_KEYS = new Set(['KeyW', 'KeyA', 'KeyS', 'KeyD', 'ShiftLeft', 'ShiftRight', 'KeyE']);
const OBSTACLE = { minX: -.575 - .18, maxX: .255 + .18, minZ: .25 - .18, maxZ: .81 + .18 };
const editable = element => !!element?.closest?.('input, textarea, select, button, [contenteditable="true"]');
const insideTable = (x, z) => x > OBSTACLE.minX && x < OBSTACLE.maxX && z > OBSTACLE.minZ && z < OBSTACLE.maxZ;

/** Local procedural locomotion for the real articulated G1 link model.
 * worldRoot is Y-up; its child robotRoot performs the native MuJoCo conversion.
 * Walking is a visual gait, not a Kimodo-generated motion or hardware control.
 */
export class RobotController {
  constructor({ worldRoot, robotRoot, camera, controls, joints, onInteract, onStatus, onManualInput, obstacles=[] }) {
    this.worldRoot = worldRoot;
    this.robotRoot = robotRoot;
    this.camera = camera;
    this.controls = controls;
    this.joints = joints;
    this.onInteract = onInteract || (() => {});
    this.onStatus = onStatus || (() => {});
    this.onManualInput = onManualInput || (() => {});
    this.remoteInput = null;
    this.obstacles = obstacles;
    this.byName = new Map(joints.map(joint => [joint.name, joint]));
    this.pelvis = robotRoot.getObjectByName('pelvis');
    this.pelvisRest = this.pelvis?.position.clone();
    this.pelvisRestQuaternion = this.pelvis?.quaternion.clone();
    this.feet = ['left_ankle_roll_link', 'right_ankle_roll_link'].map(name => robotRoot.getObjectByName(name)).filter(Boolean);
    this.keys = new Set();
    this.velocity = new THREE.Vector3();
    this.direction = new THREE.Vector3();
    this.right = new THREE.Vector3();
    this.delta = new THREE.Vector3();
    this.box = new THREE.Box3();
    this.rotation = new THREE.Quaternion();
    this.phase = 0;
    this.blend = 0;
    this._enabled = false;
    this._moving = false;
    this.lastStatus = '';
    this.canvas = controls.domElement;
    this.previousTabIndex = this.canvas?.getAttribute('tabindex');
    if (this.canvas) this.canvas.tabIndex = 0;
    this.onPointerDown = () => { if (this._enabled) this.canvas?.focus({ preventScroll: true }); };
    this.onKeyDown = event => {
      if (!this._enabled || editable(event.target) || event.ctrlKey || event.altKey || event.metaKey || !MOVEMENT_KEYS.has(event.code)) return;
      event.preventDefault();
      this.onManualInput();
      this.keys.add(event.code);
      if (event.code === 'KeyE' && !event.repeat) this.onInteract();
    };
    this.onKeyUp = event => {
      this.keys.delete(event.code);
      if (this._enabled && MOVEMENT_KEYS.has(event.code) && !editable(event.target)) event.preventDefault();
    };
    this.onBlur = () => this.keys.clear();
    this.onVisibility = () => { if (document.hidden) this.keys.clear(); };
    window.addEventListener('keydown', this.onKeyDown);
    window.addEventListener('keyup', this.onKeyUp);
    window.addEventListener('blur', this.onBlur);
    document.addEventListener('visibilitychange', this.onVisibility);
    this.canvas?.addEventListener('pointerdown', this.onPointerDown);
  }

  get enabled() { return this._enabled; }
  get moving() { return this._moving; }

  drive(forward = 0, right = 0, sprint = false) {
    this.remoteInput = { forward, right, sprint, heading: this.remoteInput?.heading ?? this.worldRoot.rotation.y };
  }

  stopDrive() {
    this.remoteInput = null;
    this.velocity.set(0, 0, 0);
    this.keys.clear();
  }

  blocked(x,z) {
    if(insideTable(x,z))return true;
    const before=this.worldRoot.position;
    for(const o of this.obstacles){
      const penetration=(px,pz)=>Math.min(px-o.minX,o.maxX-px,pz-o.minZ,o.maxZ-pz);
      const next=penetration(x,z),previous=penetration(before.x,before.z);
      // A stood-up robot starts at the calibrated cushion edge. Let it
      // leave the coarse collision box, while preventing further entry.
      if(next>0&&(previous<=0||next>=previous-1e-8))return true;
    }
    return false;
  }

  setEnabled(value) {
    const enabled = !!value;
    if (enabled === this._enabled) return;
    this._enabled = enabled;
    this.keys.clear();
    this.remoteInput = null;
    this.velocity.set(0, 0, 0);
    this._moving = false;
    this.blend = 0;
    this.lastStatus = '';
    if (this.pelvis && this.pelvisRest) {
      this.pelvis.position.copy(this.pelvisRest);
      this.pelvis.quaternion.copy(this.pelvisRestQuaternion);
    }
    if (enabled) {
      this.canvas?.focus({ preventScroll: true });
      this.updatePose(0, 0);
      this.report('Управление: WASD · Shift — быстрее · E — кофе');
    }
  }

  report(message) {
    if (message !== this.lastStatus) {
      this.lastStatus = message;
      this.onStatus(message);
    }
  }

  setJoint(name, value) {
    const joint = this.byName.get(name);
    if (!joint) return;
    joint.angle = THREE.MathUtils.clamp(value, joint.min, joint.max);
    joint.node.quaternion.copy(joint.rest).multiply(this.rotation.setFromAxisAngle(joint.axis, joint.angle));
  }

  update(dt) {
    if (!this._enabled) return;
    dt = Math.min(Math.max(dt, 0), .05);
    const forward = this.remoteInput?.forward ?? Number(this.keys.has('KeyW')) - Number(this.keys.has('KeyS'));
    const sideways = this.remoteInput?.right ?? Number(this.keys.has('KeyD')) - Number(this.keys.has('KeyA'));
    if (this.remoteInput) this.direction.set(Math.sin(this.remoteInput.heading), 0, Math.cos(this.remoteInput.heading));
    else this.camera.getWorldDirection(this.direction);
    this.direction.y = 0;
    if (this.direction.lengthSq() < .001) this.direction.set(0, 0, 1);
    this.direction.normalize();
    this.right.crossVectors(this.direction, UP).normalize();
    this.direction.multiplyScalar(forward).addScaledVector(this.right, sideways);
    const sprint = this.remoteInput?.sprint ?? (this.keys.has('ShiftLeft') || this.keys.has('ShiftRight'));
    const speed = sprint ? 2.5 : 1.5;
    if (this.direction.lengthSq()) this.direction.normalize().multiplyScalar(speed);
    this.velocity.lerp(this.direction, 1 - Math.exp(-dt * (this.direction.lengthSq() ? 7 : 10)));
    if (this.velocity.lengthSq() < .00001) this.velocity.set(0, 0, 0);

    const position = this.worldRoot.position;
    const beforeX = position.x, beforeZ = position.z;
    // Separate axes let the robot slide along the table instead of sticking.
    const proposedX = THREE.MathUtils.clamp(position.x + this.velocity.x * dt, -4, 4);
    if (!this.blocked(proposedX, position.z)) position.x = proposedX;
    const proposedZ = THREE.MathUtils.clamp(position.z + this.velocity.z * dt, -4, 4);
    if (!this.blocked(position.x, proposedZ)) position.z = proposedZ;
    this.delta.set(position.x - beforeX, 0, position.z - beforeZ);
    const actualSpeed = dt > 0 ? this.delta.length() / dt : 0;
    this._moving = actualSpeed > .03;
    if (this.velocity.lengthSq() > .003) {
      const heading = Math.atan2(this.velocity.x, this.velocity.z);
      const difference = THREE.MathUtils.euclideanModulo(heading - this.worldRoot.rotation.y + Math.PI, Math.PI * 2) - Math.PI;
      this.worldRoot.rotation.y += difference * (1 - Math.exp(-dt * 11));
    }
    this.camera.position.add(this.delta);
    this.controls.target.add(this.delta);
    this.blend = THREE.MathUtils.lerp(this.blend, Math.min(actualSpeed / speed, 1), 1 - Math.exp(-dt * 10));
    this.phase += dt * (sprint ? 13.5 : 10.5) * Math.min(actualSpeed / speed, 1);
    this.updatePose(this.blend, sprint ? 1.15 : 1);
    this.report(this._moving ? (sprint ? 'Бежит быстрее · процедурная анимация' : 'Бежит · процедурная анимация') : 'Управление: WASD · Shift — быстрее · E — кофе');
  }

  updatePose(blend, stride) {
    // Restoring every articulated link prevents a coffee pose leaking into locomotion.
    for (const joint of this.joints) this.setJoint(joint.name, 0);
    if (this.pelvis && this.pelvisRest) {
      this.pelvis.position.copy(this.pelvisRest);
      this.pelvis.quaternion.copy(this.pelvisRestQuaternion);
    }
    for (const [side, offset, sign] of [['left', 0, 1], ['right', Math.PI, -1]]) {
      const phase = this.phase + offset;
      const swing = Math.sin(phase), lift = Math.max(swing, 0);
      const hip = -.42 * swing * blend * stride;
      const knee = (.12 + .67 * lift) * blend;
      this.setJoint(`${side}_hip_pitch_joint`, hip);
      this.setJoint(`${side}_hip_roll_joint`, sign * (.045 + .015 * Math.cos(phase) * blend));
      this.setJoint(`${side}_hip_yaw_joint`, .025 * swing * blend);
      this.setJoint(`${side}_knee_joint`, knee);
      this.setJoint(`${side}_ankle_pitch_joint`, -hip - knee + .08 * blend);
      this.setJoint(`${side}_ankle_roll_joint`, -sign * .045);
      this.setJoint(`${side}_shoulder_pitch_joint`, .1 + .33 * swing * blend);
      this.setJoint(`${side}_shoulder_roll_joint`, sign * .09);
      this.setJoint(`${side}_shoulder_yaw_joint`, 0);
      this.setJoint(`${side}_elbow_joint`, .95 + .15 * lift * blend);
      this.setJoint(`${side}_wrist_pitch_joint`, -.12);
      this.setJoint(`${side}_hand_thumb_0_joint`, .2);
      this.setJoint(`${side}_hand_thumb_1_joint`, sign * .25);
      this.setJoint(`${side}_hand_thumb_2_joint`, sign * .3);
      for (const finger of ['index', 'middle']) {
        this.setJoint(`${side}_hand_${finger}_0_joint`, -sign * .25);
        this.setJoint(`${side}_hand_${finger}_1_joint`, -sign * .3);
      }
    }
    this.setJoint('waist_yaw_joint', .025 * Math.sin(this.phase) * blend);
    this.setJoint('waist_pitch_joint', .035 * blend);
    this.worldRoot.updateMatrixWorld(true);
    let lowest = Infinity;
    for (const foot of this.feet) {
      this.box.setFromObject(foot);
      lowest = Math.min(lowest, this.box.min.y);
    }
    // Native pelvis Z is vertical after the native→Y-up conversion.
    if (this.pelvis && Number.isFinite(lowest)) {
      this.pelvis.position.z -= lowest;
      this.worldRoot.updateMatrixWorld(true);
    }
  }

  dispose() {
    this.setEnabled(false);
    window.removeEventListener('keydown', this.onKeyDown);
    window.removeEventListener('keyup', this.onKeyUp);
    window.removeEventListener('blur', this.onBlur);
    document.removeEventListener('visibilitychange', this.onVisibility);
    this.canvas?.removeEventListener('pointerdown', this.onPointerDown);
    if (this.canvas) {
      if (this.previousTabIndex == null) this.canvas.removeAttribute('tabindex');
      else this.canvas.setAttribute('tabindex', this.previousTabIndex);
    }
  }
}
