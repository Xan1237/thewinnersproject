import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RobotController } from './controller.js';
import { SimulationApi } from './api-client.js';
import './style.css';

const $ = id => document.getElementById(id);
const motionLabel = () => motion ? 'движение Kimodo-G1 + IK для захвата' : 'процедурный захват';
const DURATION = 12;
let time = 0, playing = !matchMedia('(prefers-reduced-motion: reduce)').matches;
let robot, motion, metadata, grip, joints = [], bodies = new Map(), rest = new Map();
let controller, api, mode='demo', manualCoffee=false, alignment=null;
const stage = $('stage');
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.3;
stage.append(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color('#b6c6ce');
scene.fog = new THREE.Fog('#b6c6ce', 8, 22);
const camera = new THREE.PerspectiveCamera(38, innerWidth / innerHeight, 0.03, 50);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.minDistance = 0.4; controls.maxDistance = 9;
controls.maxPolarAngle = Math.PI * 0.49;
camera.position.set(-2.1, 1.6, 2.8); controls.target.set(-0.05, 0.82, 0.12);
scene.add(new THREE.HemisphereLight('#eef7ff', '#6f6355', 2.8));
const key = new THREE.DirectionalLight('#fff1df', 4);
key.position.set(-3, 5, 3); key.castShadow = true;
key.shadow.mapSize.set(2048, 2048); key.shadow.camera.left = -3; key.shadow.camera.right = 3;
key.shadow.camera.top = 3; key.shadow.camera.bottom = -3;
key.shadow.bias = -0.00015; key.shadow.normalBias = 0.005;
scene.add(key);
const rim = new THREE.DirectionalLight('#d8edff', 2.0); rim.position.set(2,3,-2); scene.add(rim);
const mat = (color, roughness = .55, metalness = 0) => new THREE.MeshStandardMaterial({color,roughness,metalness});
function mesh(geometry, material, position, parent = scene) {
  const m = new THREE.Mesh(geometry, material); m.position.set(...position); m.castShadow = true; m.receiveShadow = true; parent.add(m); return m;
}
mesh(new THREE.PlaneGeometry(100,100),mat('#b6c6ce',.85),[0,0,0]).rotation.x=-Math.PI/2;
const table = new THREE.Group(); scene.add(table);
mesh(new THREE.BoxGeometry(.83,.055,.56),mat('#7e5e46',.42),[-.16,.76,.53],table);
for (const x of [-.5,.18]) for(const z of [.32,.74]) mesh(new THREE.CylinderGeometry(.018,.021,.73,12),mat('#3b4b50',.45,.5),[x,.366,z],table);
mesh(new THREE.BoxGeometry(.76,.045,.018),mat('#3b4b50',.45,.5),[-.16,.62,.74],table);
const saucerPosition = new THREE.Vector3(-.28,.795,.38);
mesh(new THREE.CylinderGeometry(.085,.077,.012,48),mat('#d7e1e5',.25),saucerPosition.toArray());
const cup = new THREE.Group(); scene.add(cup);
const cupRest = new THREE.Vector3(-.28,.799,.38);
cup.position.copy(cupRest);
const porcelain = mat('#f1f1eb',.2);
// Open ceramic profile: actual hollow mouth instead of a capped cylinder.
const profile = [[0,0],[.035,0],[.042,.007],[.047,.085],[.043,.088],[.038,.011],[0,.011]].map(([x,y])=>new THREE.Vector2(x,y));
mesh(new THREE.LatheGeometry(profile,48),porcelain,[0,0,0],cup);
mesh(new THREE.CylinderGeometry(.04,.04,.002,40),mat('#462c1b',.18),[0,.074,0],cup);
const handle=mesh(new THREE.TorusGeometry(.026,.006,10,30),porcelain,[-.061,.048,0],cup);
handle.scale.set(.85,1.25,1);
// Fine rising steam, kept subtle and local to the cup.
const steamMat = new THREE.LineBasicMaterial({color:'#f2f5f2',transparent:true,opacity:.30});
for(let i=0;i<3;i++){
  const pts=Array.from({length:24},(_,j)=>new THREE.Vector3(.011*Math.sin(j*.45+i),.1+j*.007,.009*Math.cos(j*.4+i)));
  cup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),steamMat));
}
const nativeToWorld = new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().set(0,1,0,0,0,0,1,0,1,0,0,0,0,0,0,1));
const worldRoot=new THREE.Group();scene.add(worldRoot);
const robotRoot = new THREE.Group(); robotRoot.quaternion.copy(nativeToWorld); worldRoot.add(robotRoot);
const phases = [[0,'Перед чашкой'],[1,'Тянется к кофе'],[3.3,'Захватывает чашку'],[4,'Поднимает'],[6,'Держит кофе'],[8,'Ставит на стол'],[10,'Отпускает'],[10.8,'Возвращает руку']];
const ease = x => {x=THREE.MathUtils.clamp(x,0,1);return x*x*(3-2*x)};
const interp=(t,a,b)=>ease((t-a)/(b-a));
function phase(t){return phases.filter(([start])=>t>=start).at(-1)[1]}
function setJoint(j,angle){j.angle=THREE.MathUtils.clamp(angle,j.min,j.max);j.node.quaternion.copy(j.rest).multiply(new THREE.Quaternion().setFromAxisAngle(j.axis,j.angle))}
const wp=()=>grip.getWorldPosition(new THREE.Vector3());
function solveLinear(a,b){
  a=a.map((r,i)=>[...r,b[i]]);
  for(let c=0;c<b.length;c++){
    let p=c;for(let r=c+1;r<b.length;r++)if(Math.abs(a[r][c])>Math.abs(a[p][c]))p=r;
    [a[c],a[p]]=[a[p],a[c]];const d=a[c][c];if(Math.abs(d)<1e-10)return b.map(()=>0);
    for(let k=c;k<=b.length;k++)a[c][k]/=d;
    for(let r=0;r<b.length;r++)if(r!==c){const f=a[r][c];for(let k=c;k<=b.length;k++)a[r][k]-=f*a[c][k]}
  }return a.map(r=>r[b.length]);
}
const desiredHandQ=nativeToWorld.clone();
function armIK(target,weight=1){
  if(!grip||!joints.length||weight<=0)return;
  const before=joints.map(j=>j.angle);
  for(let iteration=0;iteration<22;iteration++){
    robotRoot.updateMatrixWorld(true);
    const p=wp(),q=grip.getWorldQuaternion(new THREE.Quaternion());
    const qe=desiredHandQ.clone().multiply(q.clone().invert()); if(qe.w<0)qe.set(-qe.x,-qe.y,-qe.z,-qe.w);
    const e=[target.x-p.x,target.y-p.y,target.z-p.z,qe.x*.16,qe.y*.16,qe.z*.16];
    if(Math.hypot(...e)<.0009)break;
    const cols=joints.map(j=>{
      const pivot=j.node.getWorldPosition(new THREE.Vector3());
      const axis=j.axis.clone().applyQuaternion(j.node.parent.getWorldQuaternion(new THREE.Quaternion()).multiply(j.rest));
      const linear=axis.clone().cross(p.clone().sub(pivot));
      return [linear.x,linear.y,linear.z,axis.x*.08,axis.y*.08,axis.z*.08];
    });
    const A=Array.from({length:6},(_,r)=>Array.from({length:6},(_,c)=>cols.reduce((v,col)=>v+col[r]*col[c],0)+(r===c?.0004:0)));
    const v=solveLinear(A,e);
    cols.forEach((col,i)=>setJoint(joints[i],joints[i].angle+THREE.MathUtils.clamp(col.reduce((sum,x,k)=>sum+x*v[k],0),-.13,.13)));
  }
  if(weight<1)joints.forEach((j,i)=>setJoint(j,THREE.MathUtils.lerp(before[i],j.angle,weight)));
  robotRoot.updateMatrixWorld(true);
}
function applyMotion(t){
  for(const [node,q] of rest)node.quaternion.copy(q);
  for(const j of allJoints){setJoint(j,0)}
  if(!motion?.mujoco_qpos)return;
  const frames=motion.mujoco_qpos, frame=Math.min(frames.length-1,t/DURATION*(frames.length-1));
  const i=Math.floor(frame),f=frame-i,a=frames[i],b=frames[Math.min(i+1,frames.length-1)];
  const names=motion.mujoco_joint_names;
  for(let k=0;k<names.length;k++){
    const name=names[k].replace(/_joint$/,'');
    const j=allJoints.find(x=>x.name===name||x.name===names[k]||x.body===name+'_link');
    if(j)setJoint(j,THREE.MathUtils.lerp(a[k+7],b[k+7],f));
  }
  const pelvis=bodies.get('pelvis');
  pelvis.position.z=THREE.MathUtils.lerp(a[2],b[2],f);
  pelvis.quaternion.set(a[4],a[5],a[6],a[3]).slerp(new THREE.Quaternion(b[4],b[5],b[6],b[3]),f);
  // Foot grounding compensates imported rest offsets while retaining generated leg rotations.
  robotRoot.updateMatrixWorld(true);
  let sole=Infinity;
  for(const side of ['left','right']){const foot=bodies.get(side+'_ankle_roll_link');if(foot)sole=Math.min(sole,new THREE.Box3().setFromObject(foot).min.y)}
  if(Number.isFinite(sole))pelvis.position.z-=sole;
}
let allJoints=[];
const graspTarget=cupRest.clone().add(new THREE.Vector3(.005,.047,-.005));
const holdTarget=new THREE.Vector3(-.25,1.055,.25);
const homeTarget=new THREE.Vector3(-.27,.88,.04);
const cupLocal=new THREE.Vector3();
function pose(t){
  if(!robot)return;
  applyMotion(t);
  // A seven-axis arm solve provides exact object contact on top of the generated gesture.
  let target=homeTarget.clone();
  if(t<1)target.lerp(graspTarget,0);
  else if(t<3.3)target.lerp(graspTarget,interp(t,1,3.3));
  else if(t<4)target.copy(graspTarget);
  else if(t<6)target.copy(graspTarget).lerp(holdTarget,interp(t,4,6));
  else if(t<8)target.copy(holdTarget);
  else if(t<10)target.copy(holdTarget).lerp(graspTarget,interp(t,8,10));
  else if(t<10.8)target.copy(graspTarget);
  else target.copy(graspTarget).lerp(homeTarget,interp(t,10.8,12));
  armIK(target);
  const close= t<3.3?0:t<4?interp(t,3.3,4):t<10?1:1-interp(t,10,10.8);
  for(const j of allJoints.filter(j=>j.name.includes('right_hand_'))){
    const thumb=j.name.includes('thumb');
    setJoint(j,(thumb?(j.name.includes('thumb_0')?.25:-.65):.75)*close);
  }
  robotRoot.updateMatrixWorld(true);
  cup.position.copy(cupRest);cup.quaternion.identity();
  if(t>=4&&t<=10){
    // Captured grasp-relative offset; world cup remains upright during the wrist correction.
    cup.position.copy(wp()).add(new THREE.Vector3(-.005,-.047,.005));
  }
  $('phase').textContent=phase(t);$('clock').textContent=`${t.toFixed(1)} / ${DURATION.toFixed(1)} с`;$('timeline').value=t;
  stage.dataset.contactError=wp().distanceTo(target).toFixed(6);stage.dataset.cup=cup.position.toArray().join(',');stage.dataset.time=t.toFixed(3);stage.dataset.motion=motion?'kimodo-g1':'procedural';
  window.demoState={time:t,phase:phase(t),cup:cup.position.toArray(),grip:wp().toArray(),contactError:wp().distanceTo(target),loaded:true,motion:!!motion,triangles:renderer.info.render.triangles};
}
async function load(){
  try{
    const [gltf,meta,mot]=await Promise.all([new GLTFLoader().loadAsync('/assets/g1/g1.glb'),fetch('/assets/g1/metadata.json').then(r=>r.json()),fetch('/assets/motion/coffee.json').then(r=>r.ok?r.json():null).catch(()=>null)]);
    metadata=meta;motion=mot;robot=gltf.scene;robotRoot.add(robot);
    robot.traverse(node=>{if(node.isMesh){node.castShadow=true;node.receiveShadow=true;}bodies.set(node.name,node);rest.set(node,node.quaternion.clone())});
    const entries=Array.isArray(meta.bodies)?meta.bodies:Object.values(meta.bodies||{});
    for(const body of entries){
      const node=bodies.get(body.name); if(!node)continue;
      for(const joint of body.joints|| (body.joint?[body.joint]:[])){
        const range=joint.range||joint.limits||[-Math.PI,Math.PI];
        allJoints.push({name:joint.name,body:body.name,node,rest:node.quaternion.clone(),axis:new THREE.Vector3(...joint.axis),min:range[0],max:range[1],angle:0});
      }
    }
    joints=allJoints.filter(j=>['right_shoulder_pitch_joint','right_shoulder_roll_joint','right_shoulder_yaw_joint','right_elbow_joint','right_wrist_roll_joint','right_wrist_pitch_joint','right_wrist_yaw_joint'].includes(j.name));
    const wrist=bodies.get('right_wrist_yaw_link');
    if(!wrist||joints.length!==7)throw Error(`Не найдена цепочка руки: ${joints.length} суставов`);
    grip=new THREE.Object3D();grip.position.set(.1465,.022,0);wrist.add(grip);
    controller=new RobotController({worldRoot,robotRoot,camera,controls,joints:allJoints,onInteract:interact,onManualInput:()=>{if(api?.active)api.cancel()},onStatus:message=>{if(mode==='control'&&!manualCoffee)$('phase').textContent=message}});
    api=new SimulationApi({
      status:message=>{$('api-status').textContent=message;$('api-connect').textContent=api.socket?'Отключить API':'Подключить API'},
      state:()=>({position:worldRoot.position.toArray(),heading:worldRoot.rotation.y,mode,coffeeActive:manualCoffee||!!alignment,cup:cup.position.toArray(),motion:motion?'kimodo-g1':'procedural',time,joints:Object.fromEntries(allJoints.map(j=>[j.name,j.angle])),limits:Object.fromEntries(allJoints.map(j=>[j.name,[j.min,j.max]])),activeCommand:api?.active?.id??null}),
      control:()=>{if(manualCoffee||alignment)throw Error('Сейчас робот занят кофе');if(mode!=='control')setMode('control')},
      drive:(forward,right,sprint)=>controller.drive(forward,right,sprint),
      stopDrive:()=>controller.stopDrive(),
      turn:heading=>{worldRoot.rotation.y=heading},
      coffee:()=>{if(worldRoot.position.length()>.75)throw Error('Подойди к кофейной станции: расстояние до [0,0,0] должно быть не больше 0.75 м');interact()},
      validateAngles:angles=>Object.entries(angles).map(([name,angle])=>{
        const joint=allJoints.find(j=>j.name===name);
        if(!joint||!Number.isFinite(angle)||angle<joint.min||angle>joint.max)throw Error(`Недопустимый угол сустава ${name}`);
        return [joint,angle];
      }),
      setAngles:angles=>{for(const [joint,angle] of angles)setJoint(joint,angle);robotRoot.updateMatrixWorld(true)},
      stop:()=>{controller.stopDrive();playing=false;if(mode==='control'){manualCoffee=false;alignment=null;controller.setEnabled(true);cup.position.copy(cupRest)}},
    });
    $('api-connect').onclick=()=>{if(api.socket)api.disconnect();else api.connect($('api-token').value)};
    $('loading').hidden=true;
    $('provenance').textContent=`Unitree G1 · ${motionLabel()} · локальная демка`;
    pose(0);window.setDemoTime=t=>{time=THREE.MathUtils.clamp(t,0,DURATION);playing=false;pose(time);renderer.render(scene,camera)};
  }catch(e){$('loading').textContent=`Не удалось загрузить демку: ${e.message}`;console.error(e)}
}
$('play').onclick=()=>{playing=!playing;$('play').textContent=playing?'Пауза':'Продолжить'};
$('replay').onclick=()=>{if(mode==='control')interact();else{time=0;playing=true;$('play').textContent='Пауза'}};
$('timeline').oninput=e=>{time=+e.target.value;playing=false;$('play').textContent='Продолжить';pose(time)};
function view(close){camera.position.set(...(close?[-1.05,1.26,1.15]:[-2.1,1.6,2.8]));controls.target.set(...(close?[-.25,.94,.3]:[-.05,.82,.12]));camera.position.add(worldRoot.position);controls.target.add(worldRoot.position);$('close').classList.toggle('selected',close);$('wide').classList.toggle('selected',!close);controls.update()}
$('wide').onclick=()=>view(false);$('close').onclick=()=>view(true);
function setMode(next){
  if(!controller)return;mode=next;manualCoffee=false;alignment=null;
  controller.setEnabled(next==='control');time=0;playing=next==='demo';
  if(next==='demo'){worldRoot.position.set(0,0,0);worldRoot.quaternion.identity()}
  cup.position.copy(cupRest);cup.quaternion.identity();
  $('manual').classList.toggle('selected',next==='control');$('demo').classList.toggle('selected',next==='demo');
  $('timeline').disabled=next==='control';$('play').disabled=next==='control';
  $('replay').textContent=next==='control'?'Взять кофе · E':'Сначала ↺';
  $('subtitle').textContent=next==='control'?'WASD — движение · Shift — быстрее · E у стола — взять кофе':'Берёт чашку. Поднимает. Возвращает на стол.';
  stage.tabIndex=0;stage.focus();view(false);
  $('provenance').textContent=next==='control'?'Unitree G1 · WASD / Shift / E · перетаскивай сцену, чтобы повернуть камеру':`Unitree G1 · ${motionLabel()} · локальная демка`;
}
function interact(){
  if(mode!=='control'||manualCoffee)return;
  if(worldRoot.position.length()>.75){$('phase').textContent='Подойди к столу и нажми E';return}
  controller.setEnabled(false);manualCoffee=true;time=0;playing=false;
  alignment={elapsed:0,position:worldRoot.position.clone(),rotation:worldRoot.quaternion.clone()};
}
$('manual').onclick=()=>{api?.cancel();setMode('control')};$('demo').onclick=()=>{api?.cancel();setMode('demo')};
addEventListener('resize',()=>{camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight)});
let previous=performance.now();
function animate(now){
  requestAnimationFrame(animate);const dt=Math.min((now-previous)/1000,.05);previous=now;
  if(robot){
    api?.beforeUpdate(dt);
    if(alignment){
      const previousPosition=worldRoot.position.clone();alignment.elapsed+=dt;
      const f=ease(alignment.elapsed/.65);worldRoot.position.copy(alignment.position).multiplyScalar(1-f);worldRoot.quaternion.copy(alignment.rotation).slerp(new THREE.Quaternion(),f);
      const delta=worldRoot.position.clone().sub(previousPosition);camera.position.add(delta);controls.target.add(delta);pose(0);$('phase').textContent='Поворачивается к чашке';
      if(f===1){alignment=null;playing=true}
    }else if(mode==='control'&&!manualCoffee){
      controller.update(dt);cup.position.copy(cupRest);cup.quaternion.identity();
      $('clock').textContent='WASD / Shift / E';stage.dataset.robotPosition=worldRoot.position.toArray().join(',');stage.dataset.mode='control';
    }else{
      if(playing){time+=dt;if(time>DURATION){if(manualCoffee){time=DURATION;playing=false;manualCoffee=false;controller.setEnabled(true)}else if($('loop').checked)time%=DURATION;else{time=DURATION;playing=false;$('play').textContent='Продолжить'}}}
      pose(time);stage.dataset.mode=mode;
    }
    api?.afterUpdate(dt);
  }controls.update();renderer.render(scene,camera);
}
load();requestAnimationFrame(animate);
