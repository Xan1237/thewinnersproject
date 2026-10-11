// Plain-English names for the recipe and for the numbers in the labeled data.
export const PARTS = {
  fetch_coffee: 'pick up a cup',
  'station:place': 'put the cup under the spout',
  'station:press': 'press the brew button',
  'station:wait': 'wait for the coffee',
  'station:take': 'take the full cup',
  'station:serve': 'put it on the serving counter',
  cheer: 'cheer',
};

/** One entry per number of the robot's sensor vector (arena/server/core.js OBSERVATION_SPEC). */
export const STATE = [
  ['x', 'position left/right in the room (m)'], ['z', 'position front/back in the room (m)'], ['heading', 'which way the robot faces (rad)'],
  ['goal_forward', 'how far ahead the next waypoint is (m)'], ['goal_right', 'how far to the right of it (m)'], ['goal_dist', 'distance to the thing it needs (m)'],
  ['holding', '1 = a cup is in its hand'], ['time_left', 'share of the time budget still left'],
  ['task_progress', 'how far through the 7 parts (0 to 1)'], ['can_use', '1 = this part needs a pick up / press'], ['must_wait', '1 = this part is waiting for the machine'], ['machine', 'its coffee spout: 0 empty, .33 cup in, .67 brewing, 1 ready'],
];
export const ACTION = [
  ['forward', 'walk forward (+) or back (-)'], ['right', 'sidestep right (+) or left (-)'], ['turn', 'turn left (+) or right (-)'], ['interact', 'above 0.5 = pick up / put down / press'],
];
