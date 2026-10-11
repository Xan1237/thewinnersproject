import { sampleSit } from '../../robot/src/sitting-motion.js';
import * as THREE from 'three';
import { G1 } from './g1.js';
import { HOLD_POSE, gaitJoints } from '../server/core.js';
import { getClip } from './animations.js';
import { kindOf, KIND_LABEL } from './kinds.js';

const STEP_SAMPLE = 0.1; // seconds between path samples (matches the sim: 5 samples per 0.5 s step)
const lerpAngle = (a, b, t) => a + (((b - a + Math.PI * 3) % (Math.PI * 2)) - Math.PI) * t;

function makeTag(text, sub, color, kind = 'frontier') {
  const c = document.createElement('canvas'); c.width = 512; c.height = 160;
  const g = c.getContext('2d'), trained = kind === 'trained';
  g.fillStyle = trained ? 'rgba(40,30,4,.92)' : 'rgba(12,16,24,.86)'; g.beginPath(); g.roundRect(4, 4, 504, 152, 18); g.fill();
  g.strokeStyle = color; g.lineWidth = trained ? 7 : 4; g.stroke();
  // what kind of decision-maker this is
  g.font = '700 22px system-ui, sans-serif';
  const badge = trained ? '★ TRAINED MODEL · 0 tokens' : kind === 'scripted' ? 'SCRIPTED TEACHER' : 'FRONTIER · via Jev';
  const w = g.measureText(badge).width + 24;
  g.fillStyle = trained ? color : '#27324a'; g.beginPath(); g.roundRect(18, 14, w, 32, 16); g.fill();
  g.fillStyle = trained ? '#241a00' : '#9fb4d8'; g.fillText(badge, 30, 38);
  g.fillStyle = color; g.font = '600 38px system-ui, sans-serif'; g.fillText(text.slice(0, 22), 22, 90);
  g.fillStyle = '#dfe6f1'; g.font = '28px system-ui, sans-serif'; g.fillText((sub || '').slice(0, 32), 22, 134);
  const tex = new THREE.CanvasTexture(c); tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

export class Actor {
  constructor(template, scene, start, { color, model, cupFactory }) {
    this.id = start.session;
    this.model = model;
    this.color = color;
    this.task = start.task;
    this.seatPlacement = start.seat_placement ?? null;
    this.kind = kindOf(model);
    this.g1 = new G1(template, { color });
    this.scene = scene;
    scene.add(this.g1.group);
    const [x, z] = start.position;
    this.keys = [{ t: 0, x, z, h: start.heading, phase: 0, blend: 0, y: start.y, holding: false, pose: null, poseT0: 0, seatTime: -1 }];
    this.clock = 0;
    this.index = 0;
    this.cur = { ...this.keys[0] };
    this.tag = new THREE.Sprite(new THREE.SpriteMaterial({ map: makeTag(model, 'ready', color, this.kind), depthTest: false, transparent: true }));
    this.tag.scale.set(1.5, 0.469, 1); this.tag.position.y = 1.8; this.tag.renderOrder = 10;
    this.g1.group.add(this.tag);
    this.tagLabels = [];     // [{t, text}] label to show once the clock reaches t
    if (this.kind === 'trained') {
      // Our own model gets a gold halo and a light beam so it can't be mistaken for a frontier model.
      const gold = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.9, side: THREE.DoubleSide, depthWrite: false });
      this.halo = new THREE.Mesh(new THREE.RingGeometry(0.5, 0.58, 64), gold);
      this.halo.rotation.x = -Math.PI / 2; this.halo.position.y = 0.012;
      this.beam = new THREE.Mesh(new THREE.CylinderGeometry(0.46, 0.46, 3.2, 40, 1, true), new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.1, side: THREE.DoubleSide, depthWrite: false }));
      this.beam.position.y = 1.6;
      this.g1.group.add(this.halo, this.beam);
    }
    this.cupFactory = cupFactory;
    this.cup = start.task === 'fetch_coffee' ? cupFactory() : null;
    if (this.cup) scene.add(this.cup);
    this.cupHome = this.cup?.position.clone();
    this.finished = null;
    this.applyPose(this.cur, 0);
  }

  get lastT() { return this.keys[this.keys.length - 1].t; }
  get lag() { return this.lastT - this.clock; }

  enqueue(ev) {
    const last = this.keys[this.keys.length - 1];
    ev.path.forEach(([x, z, h, phase, blend, seatTime = -1], i) => {
      const lastSample = i === ev.path.length - 1;
      this.keys.push({ t: last.t + STEP_SAMPLE * (i + 1), x, z, h, phase, blend, seatTime, y: ev.y, holding: lastSample ? ev.holding : last.holding, pose: null, poseT0: 0 });
    });
    // A pose (sit, jump, ...) starts when the robot has arrived; timed ones return to standing.
    const end = this.keys[this.keys.length - 1];
    if (ev.pose) {
      end.pose = ev.pose;
      end.poseT0 = last.pose === ev.pose ? last.poseT0 : end.t;
      end.poseArg = ev.pose_offset ?? last.poseArg ?? 0;
      if (ev.pose_s > 0) {
        this.keys.push({ ...end, t: end.t + ev.pose_s });
        this.keys.push({ ...end, t: end.t + ev.pose_s + 0.02, pose: null });
      }
    }
    this.tagLabels.push({ t: last.t, text: ev.label || '' });
  }

  /** The sim gave the robot a new task where it stands (a spoken order): stand up, back to work. */
  retask(ev) {
    const last = { ...this.cur, t: this.clock };
    this.keys = [last]; this.index = 0; this.tagLabels = [];
    this.finished = null;
    this.task = ev.task;
    this.seatPlacement = ev.seat_placement ?? null;
    this.ended = false;
    this.idleFor = 0;
    this.g1.ring.material.color.set(this.color);
    this.keys.push({ ...last, t: last.t + 0.05, x: ev.position[0], z: ev.position[1], h: ev.heading, y: ev.y, holding: false, pose: null, poseT0: 0, poseArg: 0, seatTime: -1 });
    if (ev.task === 'fetch_coffee' && !this.cup) { this.cup = this.cupFactory(); this.cupHome = this.cup.position.clone(); this.scene.add(this.cup); }
    this.tagLabels.push({ t: last.t, text: ev.task_description ? `new order` : '' });
  }

  finish(success) {
    this.finished = success;
    this.g1.ring.material.color.set(success ? '#5ee0a0' : '#ff6b7a');
  }

  applyPose(k, dt) {
    const g = this.g1;
    g.group.position.set(k.x, THREE.MathUtils.damp(g.group.position.y, k.y, 8, Math.max(dt, 1 / 60)), k.z);
    g.group.rotation.y = k.h;
    const sit = this.seatPlacement && k.seatTime >= 0 ? sampleSit(k.seatTime, this.seatPlacement) : null;
    const clip = sit ? null : getClip(k.pose);
    const poseT = Math.max(this.clock - (k.poseT0 ?? 0), 0);
    const joints = sit ? sit.joints : clip ? clip.joints(poseT, k) : gaitJoints(k.phase, k.blend);
    if (k.holding && !clip) Object.assign(joints, HOLD_POSE);
    g.setJoints(joints, clip?.pelvisDrop ? clip.pelvisDrop(poseT, k) : 0, clip?.root ? clip.root(poseT, k) : null, sit?.pelvisHeight);
    if (this.cup) {
      if (k.holding) { g.handWorld(this.cup.position); this.cup.position.y -= 0.045; }
      else this.cup.position.copy(this.cupHome);
    }
  }

  update(dt, speed) {
    // Catch up when the sim runs ahead of playback (e.g. a replayed history).
    const boost = this.lag > 2 ? Math.min(8, 1 + (this.lag - 2)) : 1;
    this.clock = Math.min(this.clock + dt * speed * boost, this.lastT);
    while (this.index < this.keys.length - 2 && this.keys[this.index + 1].t <= this.clock) this.index++;
    const a = this.keys[this.index], b = this.keys[Math.min(this.index + 1, this.keys.length - 1)];
    const span = b.t - a.t;
    const f = span > 0 ? THREE.MathUtils.clamp((this.clock - a.t) / span, 0, 1) : 1;
    const idle = this.clock >= this.lastT;
    const k = {
      x: THREE.MathUtils.lerp(a.x, b.x, f), z: THREE.MathUtils.lerp(a.z, b.z, f), h: lerpAngle(a.h, b.h, f),
      phase: THREE.MathUtils.lerp(a.phase, b.phase, f), blend: THREE.MathUtils.lerp(a.blend, b.blend, f), y: b.y,
      seatTime: b.seatTime >= 0 ? (a.seatTime >= 0 ? THREE.MathUtils.lerp(a.seatTime, b.seatTime, f) : (f > 0 ? b.seatTime * f : -1)) : -1,
      holding: f >= 1 ? b.holding : a.holding, pose: f >= 1 ? b.pose : a.pose, poseT0: f >= 1 ? b.poseT0 : a.poseT0, poseArg: f >= 1 ? b.poseArg : a.poseArg,
    };
    if (idle) { k.blend = THREE.MathUtils.damp(this.cur.blend, 0, 6, dt); k.phase = this.cur.phase; }
    this.cur = k;
    this.applyPose(k, dt);
    if (this.halo) { const p = (performance.now() / 900) % 1; this.halo.scale.setScalar(1 + 0.35 * p); this.halo.material.opacity = 0.9 * (1 - p); this.beam.material.opacity = 0.08 + 0.05 * Math.sin(performance.now() / 400); }
    while (this.tagLabels.length && this.tagLabels[0].t <= this.clock) {
      const { text } = this.tagLabels.shift();
      this.tag.material.map.dispose();
      this.tag.material.map = makeTag(this.model, text || '…', this.color, this.kind);
    }
    if (this.index > 40) { this.keys.splice(0, this.index - 1); this.index = 1; }
  }

  dispose() {
    this.scene.remove(this.g1.group);
    if (this.cup) this.scene.remove(this.cup);
  }
}
