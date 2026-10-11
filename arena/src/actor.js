import * as THREE from 'three';
import { G1 } from './g1.js';
import { HOLD_POSE, gaitJoints } from '../server/core.js';
import { getClip } from './animations.js';

const STEP_SAMPLE = 0.1; // seconds between path samples (matches the sim: 5 samples per 0.5 s step)
const lerpAngle = (a, b, t) => a + (((b - a + Math.PI * 3) % (Math.PI * 2)) - Math.PI) * t;

function makeTag(text, sub, color) {
  const c = document.createElement('canvas'); c.width = 512; c.height = 128;
  const g = c.getContext('2d');
  g.fillStyle = 'rgba(12,16,24,.86)'; g.beginPath(); g.roundRect(4, 4, 504, 120, 18); g.fill();
  g.strokeStyle = color; g.lineWidth = 4; g.stroke();
  g.fillStyle = color; g.font = '600 38px system-ui, sans-serif'; g.fillText(text.slice(0, 22), 22, 52);
  g.fillStyle = '#dfe6f1'; g.font = '30px system-ui, sans-serif'; g.fillText((sub || '').slice(0, 30), 22, 100);
  const tex = new THREE.CanvasTexture(c); tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

export class Actor {
  constructor(template, scene, start, { color, model, cupFactory }) {
    this.id = start.session;
    this.model = model;
    this.color = color;
    this.g1 = new G1(template, { color });
    this.scene = scene;
    scene.add(this.g1.group);
    const [x, z] = start.position;
    this.keys = [{ t: 0, x, z, h: start.heading, phase: 0, blend: 0, y: start.y, holding: false }];
    this.clock = 0;
    this.index = 0;
    this.cur = { ...this.keys[0] };
    this.tag = new THREE.Sprite(new THREE.SpriteMaterial({ map: makeTag(model, 'ready', color), depthTest: false, transparent: true }));
    this.tag.scale.set(1.5, 0.375, 1); this.tag.position.y = 1.75; this.tag.renderOrder = 10;
    this.g1.group.add(this.tag);
    this.tagLabels = [];     // [{t, text}] label to show once the clock reaches t
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
    ev.path.forEach(([x, z, h, phase, blend], i) => {
      const lastSample = i === ev.path.length - 1;
      this.keys.push({ t: last.t + STEP_SAMPLE * (i + 1), x, z, h, phase, blend, y: ev.y, holding: lastSample ? ev.holding : last.holding, pose: ev.pose ?? null });
    });
    this.tagLabels.push({ t: last.t, text: ev.label || '' });
  }

  finish(success) {
    this.finished = success;
    this.g1.ring.material.color.set(success ? '#5ee0a0' : '#ff6b7a');
  }

  applyPose(k, dt) {
    const g = this.g1;
    g.group.position.set(k.x, THREE.MathUtils.damp(g.group.position.y, k.y, 8, Math.max(dt, 1 / 60)), k.z);
    g.group.rotation.y = k.h;
    const clip = getClip(k.pose);
    const joints = clip ? clip.joints(this.clock, k) : gaitJoints(k.phase, k.blend);
    if (k.holding && !clip) Object.assign(joints, HOLD_POSE);
    g.setJoints(joints, clip?.pelvisDrop ? clip.pelvisDrop(this.clock, k) : 0);
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
      holding: f >= 1 ? b.holding : a.holding, pose: f >= 1 ? b.pose : a.pose,
    };
    if (idle) { k.blend = THREE.MathUtils.damp(this.cur.blend, 0, 6, dt); k.phase = this.cur.phase; }
    this.cur = k;
    this.applyPose(k, dt);
    while (this.tagLabels.length && this.tagLabels[0].t <= this.clock) {
      const { text } = this.tagLabels.shift();
      this.tag.material.map.dispose();
      this.tag.material.map = makeTag(this.model, text || '…', this.color);
    }
    if (this.index > 40) { this.keys.splice(0, this.index - 1); this.index = 1; }
  }

  dispose() {
    this.scene.remove(this.g1.group);
    if (this.cup) this.scene.remove(this.cup);
  }
}
