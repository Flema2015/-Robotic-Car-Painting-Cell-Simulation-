# Robotic Car Painting Cell Simulation

**Candidate:** Emanuel Hamui
**Software:** Webots R2025a, Python, NumPy, and IKPy
**Robot:** ABB IRB 4600-40
**Vehicle:** BMW X5 model

This project demonstrates a simulated robotic painting cell, including an industrial robot with a spray-gun end-effector, a vehicle, raster painting paths, inverse kinematics, approximate collision checking, and motion around selected vehicle surfaces.

## Project files

- `Webots-project/worlds/_Robotic Car Painting SIM.wbt` — main simulation world.
- `Webots-project/controllers/Painting_car/` — robot controller, surface/path model, collision checks, and focused tests.
- `Webots-project/controllers/scan_rig/` — RGB-D scan capture and image-processing utilities used during surface calibration.
- [`Technical_Report.pdf`](Technical_Report.pdf) — short technical report and captured simulation result.
- [`docs/Technical_Report.html`](docs/Technical_Report.html) — editable source for the report.
- [`docs/Demo_Guide.md`](docs/Demo_Guide.md) — suggested demo sequence and narration.

## Run the simulation

1. Install Webots R2025a and make sure the Python environment used by the controller has `numpy` and `ikpy`.
2. Open `Webots-project/worlds/_Robotic Car Painting SIM.wbt` in Webots.
3. Press **Reset** and then **Run**.

The controller is currently configured for `paint_test`: it moves the arm slowly over the hood from station indices `(2, 3, 1, 4)` and places blue **visual markers** at reached points. It does not change the car texture or model a real paint process. The Supervisor teleports the robot base between stations; this is not physical base travel.

To inspect the planned paths without moving the arm, set `RUN_MODE = 'dry_run'` in `Painting_car.py`. The `motion_test` mode moves the arm without creating paint marks. Save a copy of the controller before changing modes or test settings.

## Scope and known limitations

- The hood path is the surface demonstrated in the captured run. The roof and side are modeled, but the supplied result does not establish their successful painting.
- The latest captured hood run reports 131/154 path waypoints covered by the simulated 7.5 cm marker footprint (85.1%); 23 remain uncovered. These are waypoint/footprint figures, not physical area coverage.
- Vehicle collision geometry uses conservative approximate boxes rather than exact mesh colliders. The controller retains a 10 cm clearance threshold and a 0.30 rad/s speed limit in the current test mode.
- This is a simulation prototype, not a production-ready painting system. It does not model paint flow, overspray, coating thickness, surface finish, or physically moving the robot base between stations.

See the technical report for the method, measured result, validation notes, and limitations.

## Validation

The focused controller/model test set contains 17 passing tests, and the Python files compile. A full test discovery currently encounters an import error in the exploratory `test_reach_study.py`: it requests functions not exported by `reach_study.py`, which also starts a lengthy study when imported. That exploratory test was not used as evidence for the painting result.

## Submission details to complete

- **Email:** [hamuiemanuel@gmail.com]
- **GitHub link:** [https://github.com/Flema2015/-Robotic-Car-Painting-Cell-Simulation-]
- **Google Drive link:** [https://drive.google.com/drive/folders/1MGfIITwk-NvkDfv1DSJd3E6cSEu3R65R]


