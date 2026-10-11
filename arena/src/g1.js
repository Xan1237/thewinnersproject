import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

// Native MuJoCo frame (X forward, Z up) -> three.js world (Y up, heading 0 faces +Z).
const nativeToWorld = new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().set(0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1));

export async function loadG1Template(base = '/g1') {
  const [gltf, meta] = await Promise.all([new GLTFLoader().loadAsync(`${base}/g1.glb`), fetch(`${base}/metadata.json`).then(r => r.json())]);
  return { scene: gltf.scene, meta };
}

/** One articulated Unitree G1 (rigid links, no skinning) driven by joint angles. */
export class G1 {
  constructor(template, { color = '#7cc4ff' } = {}) {
    this.group = new THREE.Group();            // world position + heading
    this.root = new THREE.Group();             // native -> world conversion
    this.root.quaternion.copy(nativeToWorld);
    this.group.add(this.root);
    this.model = template.scene.clone(true);
    this.root.add(this.model);
    this.bodies = new Map();
    this.model.traverse(n => {
      if (n.isMesh) { n.castShadow = true; n.receiveShadow = true; }
      this.bodies.set(n.name, n);
    });
    this.joints = new Map();
    const entries = Array.isArray(template.meta.bodies) ? template.meta.bodies : Object.values(template.meta.bodies || {});
    for (const body of entries) {
      const node = this.bodies.get(body.name);
      if (!node) continue;
      for (const j of body.joints || (body.joint ? [body.joint] : [])) {
        const range = j.range || j.limits || [-Math.PI, Math.PI];
        this.joints.set(j.name, { node, rest: node.quaternion.clone(), axis: new THREE.Vector3(...j.axis), min: range[0], max: range[1] });
      }
    }
    this.pelvis = this.bodies.get('pelvis');
    this.pelvisRest = this.pelvis.position.clone();
    this.pelvisRestQ = this.pelvis.quaternion.clone();
    this.feet = ['left_ankle_roll_link', 'right_ankle_roll_link'].map(n => this.bodies.get(n)).filter(Boolean);
    this.hand = this.bodies.get('right_wrist_yaw_link');
    this.box = new THREE.Box3();
    this.q = new THREE.Quaternion();
    this.ring = new THREE.Mesh(new THREE.RingGeometry(0.34, 0.4, 48), new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.9, side: THREE.DoubleSide }));
    this.ring.rotation.x = -Math.PI / 2;
    this.ring.position.y = 0.01;
    this.group.add(this.ring);
  }

  setJoints(angles, pelvisDrop = 0) {
    for (const [name, v] of this.joints) {
      const a = angles[name] ?? 0;
      const c = Math.min(Math.max(a, v.min), v.max);
      v.node.quaternion.copy(v.rest).multiply(this.q.setFromAxisAngle(v.axis, c));
    }
    this.pelvis.position.copy(this.pelvisRest);
    this.pelvis.quaternion.copy(this.pelvisRestQ);
    this.group.updateMatrixWorld(true);
    let lowest = Infinity;
    for (const foot of this.feet) { this.box.setFromObject(foot); lowest = Math.min(lowest, this.box.min.y); }
    // Keep the soles on the ground: native pelvis Z is vertical.
    if (Number.isFinite(lowest)) this.pelvis.position.z -= lowest - this.group.position.y;
    this.pelvis.position.z -= pelvisDrop;
  }

  handWorld(out = new THREE.Vector3()) {
    this.hand.updateWorldMatrix(true, false);
    return this.hand.localToWorld(out.set(0.1465, 0.022, 0));
  }
}
