import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';
import { loadG1Template } from './g1.js';
import { Actor } from './actor.js';
import { Panel } from './panel.js';
import { connectEvents } from './events.js';
import './style.css';

const $ = id => document.getElementById(id);
const NO_GL = new URLSearchParams(location.search).has('nogl'); // dev: skip 3D rendering (panel-only checks on machines without a GPU)
const stage = $('stage');
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.15;
stage.prepend(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color('#07060b');
const camera = new THREE.PerspectiveCamera(45, 1, 0.05, 80);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.maxPolarAngle = Math.PI * 0.495;
controls.minDistance = 0.8; controls.maxDistance = 30;

const VIEWS = {
  overview: { pos: [0, 11.5, 10.5], target: [0, 0.4, -0.6] },
  stage: { pos: [0.4, 2.0, 1.6], target: [0.4, 1.2, -4.0] },
};
function setView(name) {
  document.querySelectorAll('.views button').forEach(b => b.classList.toggle('selected', b.dataset.view === name));
  view = name;
  if (VIEWS[name]) { camera.position.set(...VIEWS[name].pos); controls.target.set(...VIEWS[name].target); }
}
let view = 'overview';
document.querySelectorAll('.views button').forEach(b => b.onclick = () => setView(b.dataset.view));
let speed = 1;
$('speed').onchange = e => { speed = Number(e.target.value); };

function resize() {
  const w = stage.clientWidth, h = stage.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h; camera.updateProjectionMatrix();
}
addEventListener('resize', resize);

// ---- lights (the room GLB is exported without lights)
scene.add(new THREE.HemisphereLight('#fff0e0', '#3a2f45', 1.3));
const sun = new THREE.DirectionalLight('#ffffff', 1.4);
sun.position.set(2, 12, 6);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
Object.assign(sun.shadow.camera, { left: -9, right: 9, top: 9, bottom: -9, near: 1, far: 30 });
sun.shadow.bias = -0.0004; sun.shadow.normalBias = 0.03;
scene.add(sun);
for (const [c, i, x, y, z] of [
  ['#ff3fd0', 60, -2.0, 2.6, -3.0], ['#8a5cff', 60, 3.0, 2.6, -3.0], ['#38d8ff', 40, 0.4, 2.0, -5.2],
  ['#7fe3ff', 50, 6.4, 2.0, -1.0], ['#7fe3ff', 50, -6.4, 2.0, -1.0],
  ['#fff0dc', 70, 0, 3.4, 2.5], ['#fff0dc', 50, -4, 3.4, 3.5], ['#fff0dc', 50, 4, 3.4, -1.0],
]) { const l = new THREE.PointLight(c, i, 14, 2); l.position.set(x, y, z); scene.add(l); }

// ---- room
const loader = new GLTFLoader().setMeshoptDecoder(MeshoptDecoder);
let manifest;
async function loadRoom() {
  const [gltf, man] = await Promise.all([loader.loadAsync('/room/room.glb'), fetch('/room/manifest.json').then(r => r.json())]);
  manifest = man;
  const slide = new THREE.TextureLoader().load('/room/slide.jpg'); slide.colorSpace = THREE.SRGBColorSpace;
  gltf.scene.traverse(n => {
    if (n.name === 'Ceiling' || n.name === 'WallSouth') n.visible = false;  // keep the view into the room open
    if (n.isMesh) {
      n.receiveShadow = true;
      n.castShadow = false;  // only the robots cast shadows (keeps the shadow pass cheap)
      const m = n.material;
      if (n.name === 'Slide' || (m && m.name === 'ScreenSurface')) { n.material = new THREE.MeshBasicMaterial({ map: slide }); n.castShadow = false; }
      if (m && m.transparent) m.depthWrite = false;
    }
  });
  scene.add(gltf.scene);
  // Wall glass and sky planes should not cast shadows; the screen shows the slide.
  return gltf;
}

function makeCup() {
  const porcelain = new THREE.MeshStandardMaterial({ color: '#f1f1eb', roughness: 0.2 });
  const profile = [[0, 0], [.035, 0], [.042, .007], [.047, .085], [.043, .088], [.038, .011], [0, .011]].map(([x, y]) => new THREE.Vector2(x, y));
  const cup = new THREE.Mesh(new THREE.LatheGeometry(profile, 32), porcelain);
  cup.scale.setScalar(1.3);
  const coffee = new THREE.Mesh(new THREE.CylinderGeometry(.04, .04, .002, 24), new THREE.MeshStandardMaterial({ color: '#462c1b', roughness: 0.18 }));
  coffee.position.y = 0.074; cup.add(coffee);
  cup.castShadow = true;
  const c = manifest.objects.find(o => o.id === 'coffee_cup');
  cup.position.set(c.x, 0.745, c.z);
  return cup;
}

// ---- state
const panel = new Panel();
const actors = new Map();
let template = null;
const pending = [];

function handle(ev) {
  panel.onEvent(ev);
  if (!template) { pending.push(ev); return; }
  applyActor(ev);
}
function applyActor(ev) {
  if (ev.type === 'session_started') {
    const prev = actors.get(ev.session); if (prev) prev.dispose();
    actors.set(ev.session, new Actor(template, scene, ev, { color: panel.color(ev.model), model: ev.model, cupFactory: makeCup }));
  } else if (ev.type === 'step') {
    actors.get(ev.session)?.enqueue(ev);
  } else if (ev.type === 'session_finished') {
    actors.get(ev.session)?.finish(ev.success);
  } else if (ev.type === 'reset') {
    for (const a of actors.values()) a.dispose();
    actors.clear();
  }
}

const clock = new THREE.Clock();
function frame() {
  const dt = Math.min(clock.getDelta(), 0.1);
  for (const [id, a] of actors) {
    a.update(dt, speed);
    // Finished runs linger for a few seconds, then leave the room.
    if (a.finished != null && a.lag <= 0) { a.idleFor = (a.idleFor ?? 0) + dt; if (a.idleFor > 6) { a.dispose(); actors.delete(id); } }
  }
  if (view === 'follow') {
    const a = [...actors.values()].filter(x => x.finished == null).at(-1) || [...actors.values()].at(-1);
    if (a) {
      const p = a.g1.group.position;
      controls.target.lerp(new THREE.Vector3(p.x, 1.0, p.z), 0.1);
      const back = new THREE.Vector3(-Math.sin(a.cur.h), 0, -Math.cos(a.cur.h)).multiplyScalar(3.2);
      camera.position.lerp(new THREE.Vector3(p.x + back.x, 2.2, p.z + back.z), 0.05);
    }
  }
  controls.update();
  if (!NO_GL) renderer.render(scene, camera);
  requestAnimationFrame(frame);
}

async function main() {
  resize();
  setView('overview');
  connectEvents(handle, ok => { const c = $('connection'); c.textContent = ok ? 'sim connected' : 'sim offline'; c.className = `conn ${ok ? 'on' : 'off'}`; });
  const [, tpl] = await Promise.all([loadRoom(), loadG1Template().catch(e => { console.error(e); return null; })]);
  template = tpl;
  if (!tpl) $('loading').textContent = 'G1 model missing. Run: npm run assets';
  else $('loading').hidden = true;
  if (tpl) while (pending.length) applyActor(pending.shift());
  window.arena = { scene, camera, controls, actors, panel, setView, renderer, handle };
  frame();
}
main().catch(e => { console.error(e); $('loading').textContent = `Failed to start: ${e.message}`; });
