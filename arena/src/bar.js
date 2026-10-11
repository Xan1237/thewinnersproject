// The coffee line: two long tables where the audience chairs used to be, with one coffee machine
// and one serving spot per robot. The sim says what state each spot is in (see arena/server/core.js);
// this file only draws it.
import * as THREE from 'three';
import { BAR } from '../server/core.js';

const TOP = 0.74;                     // table top height of the cloned GLB table
const steel = new THREE.MeshStandardMaterial({ color: '#2b2f38', roughness: 0.35, metalness: 0.7 });
const chrome = new THREE.MeshStandardMaterial({ color: '#b9c0cc', roughness: 0.2, metalness: 0.9 });
const porcelain = new THREE.MeshStandardMaterial({ color: '#f4f2ea', roughness: 0.25 });
const LAMP = { empty: '#3a3f4a', cup: '#ffd24d', brewing: '#ff8a2b', ready: '#43e08a' };

export function sign(text, color = '#ffffff') {
  const c = document.createElement('canvas'); c.width = 512; c.height = 96;
  const g = c.getContext('2d');
  g.fillStyle = 'rgba(10,14,22,.82)'; g.beginPath(); g.roundRect(4, 4, 504, 88, 18); g.fill();
  g.strokeStyle = color; g.lineWidth = 3; g.stroke();
  g.fillStyle = color; g.font = '600 40px system-ui, sans-serif'; g.textAlign = 'center'; g.textBaseline = 'middle';
  g.fillText(text, 256, 50);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace;
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: t, transparent: true, depthTest: false }));
  s.scale.set(1.9, 0.356, 1); s.renderOrder = 9;
  return s;
}

function cupMesh() {
  const profile = [[0, 0], [.035, 0], [.042, .007], [.047, .085], [.043, .088], [.038, .011], [0, .011]].map(([x, y]) => new THREE.Vector2(x, y));
  const cup = new THREE.Mesh(new THREE.LatheGeometry(profile, 24), porcelain);
  cup.scale.setScalar(1.5);
  const coffee = new THREE.Mesh(new THREE.CylinderGeometry(.041, .041, .004, 20), new THREE.MeshStandardMaterial({ color: '#4a2d1a', roughness: 0.2 }));
  coffee.position.y = 0.074; cup.add(coffee);
  cup.userData.coffee = coffee;
  cup.castShadow = true;
  return cup;
}

export class CoffeeLine {
  constructor(scene, room) {
    this.slots = [];
    const cx = BAR.x0 + BAR.dx * (BAR.slots - 1) / 2;
    // Tables: clones of the room's own folding table, two per bar.
    const table = room.getObjectByName('Table');
    if (table) for (const z of [BAR.machineZ, BAR.serveZ]) for (const dx of [-0.9, 0.9]) {
      const t = table.clone(true); t.position.set(cx + dx, 0, z); scene.add(t);
    }
    const a = sign('COFFEE MACHINES', '#ffcf6b'); a.position.set(cx, 1.45, BAR.machineZ + 1.0); scene.add(a);
    const b = sign('SERVING COUNTER', '#7fe3ff'); b.position.set(cx, 1.45, BAR.serveZ - 0.9); scene.add(b);
    const c = sign('CUP TABLE', '#f1f1eb'); c.position.set(4.6, 1.45, 1.4); scene.add(c);
    for (const [x, z] of [[4.62, 1.3], [4.95, 1.45], [5.25, 1.3], [4.62, 1.55], [5.25, 1.55]]) { const cup = cupMesh(); cup.position.set(x, TOP, z); scene.add(cup); }
    for (let i = 0; i < BAR.slots; i++) this.slots.push(this.makeSlot(scene, BAR.x0 + BAR.dx * i));
  }

  makeSlot(scene, x) {
    const g = new THREE.Group(); scene.add(g);
    // machine: a small espresso unit at the back of the table, the cup stands in front of it
    const mz = BAR.machineZ;
    const body = new THREE.Mesh(new THREE.BoxGeometry(0.3, 0.34, 0.2), steel); body.position.set(x, TOP + 0.17, mz + 0.14);
    const head = new THREE.Mesh(new THREE.BoxGeometry(0.3, 0.07, 0.3), steel); head.position.set(x, TOP + 0.3, mz + 0.04);
    const spout = new THREE.Mesh(new THREE.CylinderGeometry(0.018, 0.018, 0.06, 12), chrome); spout.position.set(x, TOP + 0.245, mz - 0.07);
    const lampMat = new THREE.MeshBasicMaterial({ color: LAMP.empty });
    const lamp = new THREE.Mesh(new THREE.SphereGeometry(0.028, 16, 12), lampMat); lamp.position.set(x + 0.09, TOP + 0.3, mz - 0.115);
    const ring = new THREE.Mesh(new THREE.RingGeometry(0.075, 0.095, 40), new THREE.MeshBasicMaterial({ color: '#888', side: THREE.DoubleSide }));
    ring.rotation.x = -Math.PI / 2; ring.position.set(x, TOP + 0.004, mz - 0.12);
    const cup = cupMesh(); cup.position.set(x, TOP, mz - 0.12); cup.visible = false;
    const steam = [0, 1, 2].map(() => { const s = new THREE.Mesh(new THREE.SphereGeometry(0.02, 8, 6), new THREE.MeshBasicMaterial({ color: '#fff', transparent: true, opacity: 0 })); s.position.set(x, TOP + 0.12, mz - 0.12); return s; });
    // serving spot
    const sz = BAR.serveZ;
    const sring = new THREE.Mesh(new THREE.RingGeometry(0.095, 0.125, 40), new THREE.MeshBasicMaterial({ color: '#888', side: THREE.DoubleSide }));
    sring.rotation.x = -Math.PI / 2; sring.position.set(x, TOP + 0.004, sz);
    const served = cupMesh(); served.position.set(x, TOP, sz); served.visible = false;
    g.add(body, head, spout, lamp, ring, cup, ...steam, sring, served);
    return { x, lampMat, ring, sring, cup, served, steam, state: 'empty', brewT: 0 };
  }

  /** Paint a robot's colour on its own spot. */
  claim(slot, color) {
    const s = this.slots[slot % this.slots.length]; if (!s) return;
    s.ring.material.color.set(color); s.sring.material.color.set(color);
  }

  set(slot, bar) {
    const s = this.slots[slot % this.slots.length]; if (!s) return;
    s.state = bar.machine;
    s.cup.visible = bar.machine !== 'empty';
    s.lampMat.color.set(LAMP[bar.machine] ?? LAMP.empty);
    s.cup.userData.coffee.position.y = bar.machine === 'ready' ? 0.082 : bar.machine === 'brewing' ? 0.03 + 0.05 * (1 - Math.min(1, bar.left / 8)) : 0.011;
    s.served.visible = !!bar.served;
    if (bar.served) s.served.userData.coffee.position.y = 0.082;
  }

  update(now) {
    for (const s of this.slots) {
      const on = s.state === 'brewing' || s.state === 'ready';
      s.steam.forEach((m, i) => {
        const p = ((now / 1400) + i / 3) % 1;
        m.position.y = TOP + 0.12 + p * 0.22; m.scale.setScalar(0.6 + p * 1.6);
        m.material.opacity = on ? (s.state === 'brewing' ? 0.5 : 0.22) * (1 - p) : 0;
      });
      if (s.state === 'brewing') s.lampMat.color.set(Math.sin(now / 120) > 0 ? LAMP.brewing : '#ffd24d');
    }
  }
}
