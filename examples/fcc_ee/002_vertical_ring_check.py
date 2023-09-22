import numpy as np
import xtrack as xt


tilt_machine_by_90_degrees = True
wiggler_on = True
vertical_orbit_distortion = False
check_against_tracking = True

line = xt.Line.from_json('fccee_h_thin.json')
line.build_tracker()

print('Done building tracker')

if wiggler_on:
    line.vars['on_wiggler_v'] = 0.4

if vertical_orbit_distortion:
    line['mwi.e5rg'].ksl[0] = 2e-7

# Make sure there is no vertical bend nor skew element
for ee in line.elements:
    if isinstance(ee, xt.Multipole):
        assert np.all(ee.ksl[1:] == 0)

if tilt_machine_by_90_degrees:

    tw_before_tilt = line.twiss()

    # Bring the machine to the vertical plane
    for ee in line.elements:
        if isinstance(ee, xt.Multipole):
            knl = ee.knl.copy()
            ksl = ee.ksl.copy()
            hxl = ee.hxl
            hyl = ee.hyl

            ee.hxl = -hyl
            ee.hyl = -hxl

            ee.knl[0] = -ksl[0]
            ee.ksl[0] = -knl[0]
            if len(knl) > 1:
                ee.knl[1] = -knl[1]
                ee.ksl[1] = 0
            if len(knl) > 2:
                ee.knl[2] = 0
                ee.ksl[2] = knl[2]

        if isinstance(ee, xt.DipoleEdge):
            ee._r21, ee._r43 = ee._r43, ee._r21

    tw_after_tilt = line.twiss()

    assert np.isclose(tw_after_tilt.qy, tw_before_tilt.qx, rtol=0, atol=1e-8)
    assert np.isclose(tw_after_tilt.qx, tw_before_tilt.qy, rtol=0, atol=1e-8)
    assert np.isclose(tw_after_tilt.dqy, tw_before_tilt.dqx, rtol=0, atol=1e-4)
    assert np.isclose(tw_after_tilt.dqx, tw_before_tilt.dqy, rtol=0, atol=1e-4)

    assert np.allclose(tw_after_tilt.bety, tw_before_tilt.betx, rtol=1e-5, atol=0)
    assert np.allclose(tw_after_tilt.betx, tw_before_tilt.bety, rtol=1e-5, atol=0)

    assert np.allclose(tw_after_tilt.y, tw_before_tilt.x, rtol=0, atol=1e-9)
    assert np.allclose(tw_after_tilt.x, tw_before_tilt.y, rtol=0, atol=1e-9)

    assert np.allclose(tw_after_tilt.dy, tw_before_tilt.dx, rtol=0, atol=5e-6)
    assert np.allclose(tw_after_tilt.dx, tw_before_tilt.dy, rtol=0, atol=5e-6)

tw_no_rad = line.twiss(method='4d')

line.configure_radiation(model='mean')
line.compensate_radiation_energy_loss()

tw_rad = line.twiss(eneloss_and_damping=True)
ex = tw_rad.eq_gemitt_x
ey = tw_rad.eq_gemitt_y
ez = tw_rad.eq_gemitt_zeta

# for regression testing
checked = False
if not tilt_machine_by_90_degrees and not vertical_orbit_distortion and not wiggler_on:
    assert np.isclose(ex, 6.9884e-10, atol=0,     rtol=1e-4)
    assert np.isclose(ey, 0,          atol=1e-14, rtol=0)
    assert np.isclose(ez, 3.5634e-6,  atol=0,     rtol=1e-4)
    checked = True
elif tilt_machine_by_90_degrees and not vertical_orbit_distortion and not wiggler_on:
    assert np.isclose(ex, 0,          atol=1e-14, rtol=0)
    assert np.isclose(ey, 6.9884e-10, atol=0,     rtol=1e-4)
    assert np.isclose(ez, 3.5634e-6,  atol=0,     rtol=1e-4)
    checked = True
elif not tilt_machine_by_90_degrees and not vertical_orbit_distortion and wiggler_on:
    assert np.isclose(ex, 6.9253e-10, atol=0,     rtol=1e-4)
    assert np.isclose(ey, 1.7110e-12, atol=0,     rtol=1e-4)
    assert np.isclose(ez, 3.8202e-6,  atol=0,     rtol=1e-4)
    checked = True
elif tilt_machine_by_90_degrees and not vertical_orbit_distortion and wiggler_on:
    assert np.isclose(ex, 1.7112e-12, atol=0,     rtol=1e-4)
    assert np.isclose(ey, 6.9253e-10, atol=0,     rtol=1e-4)
    assert np.isclose(ez, 3.8202e-6,  atol=0,     rtol=1e-4)
    checked = True
elif not tilt_machine_by_90_degrees and vertical_orbit_distortion and not wiggler_on:
    assert np.isclose(ex, 6.9869e-10, atol=0,     rtol=1e-4)
    assert np.isclose(ey, 2.4756e-12, atol=0,     rtol=1e-4)
    assert np.isclose(ez, 3.5404e-6,  atol=0,     rtol=1e-4)
    checked = True
elif tilt_machine_by_90_degrees and vertical_orbit_distortion and not wiggler_on:
    assert np.isclose(ex, 2.4711e-12, atol=0,     rtol=1e-4)
    assert np.isclose(ey, 6.9869e-10, atol=0,     rtol=1e-4)
    assert np.isclose(ez, 3.5404e-6,  atol=0,     rtol=1e-4)
    checked = True
else:
    raise ValueError('Unknown configuration')

assert checked

if check_against_tracking:

    line.configure_radiation(model='quantum')
    p = line.build_particles(num_particles=30)
    line.track(p, num_turns=400, turn_by_turn_monitor=True, time=True)
    mon = line.record_last_track
    print(f'Tracking time: {line.time_last_track}')

    sigma_x_eq = float(np.sqrt(ex * tw_rad.betx[0] + ey * tw_rad.betx2[0] + (np.std(p.delta) * tw_rad.dx[0])**2))
    sigma_y_eq = float(np.sqrt(ex * tw_rad.bety1[0] + ey * tw_rad.bety[0] + (np.std(p.delta) * tw_rad.dy[0])**2))
    sigma_zeta_eq = float(np.sqrt(ez * tw_rad.betz0))

    sigma_x_track = np.std(mon.x, axis=0)[-200:]
    sigma_y_track = np.std(mon.y, axis=0)[-200:]
    sigma_zeta_track = np.std(mon.zeta, axis=0)[-200:]

    if sigma_x_eq > 1e-8:
        assert np.min(np.abs(sigma_x_track/sigma_x_eq - 1.)) < 0.1
    if sigma_y_eq > 1e-8:
        assert np.min(np.abs(sigma_y_track/sigma_y_eq - 1.)) < 0.1
    assert np.min(np.abs(sigma_zeta_track/sigma_zeta_eq - 1.)) < 0.1

    assert np.isclose(sigma_x_eq, np.mean(sigma_x_track), rtol=0.3, atol=1e-9)
    assert np.isclose(sigma_y_eq, np.mean(sigma_y_track), rtol=0.3, atol=1e-9)
    assert np.isclose(sigma_zeta_eq, np.mean(sigma_zeta_track), rtol=0.3, atol=1e-9)


    import matplotlib.pyplot as plt
    plt.close('all')
    fig = plt.figure(1, figsize=(6.4, 4.8*1.3))
    spx = fig. add_subplot(3, 1, 1)
    spx.plot(np.std(mon.x, axis=0), label='track')
    spx.axhline(
        np.sqrt(ex * tw_rad.betx[0] + ey * tw_rad.betx2[0] + (np.std(p.delta) * tw_rad.dx[0])**2),
        color='red', label='twiss')
    spx.legend(loc='lower right')
    spx.set_ylabel(r'$\sigma_{x}$ [m]')

    spy = fig. add_subplot(3, 1, 2, sharex=spx)
    spy.plot(np.std(mon.y, axis=0), label='track')
    spy.axhline(
        np.sqrt(ex * tw_rad.bety1[0] + ey * tw_rad.bety[0] + (np.std(p.delta) * tw_rad.dy[0])**2),
        color='red', label='twiss')
    spx.set_ylabel(r'$\sigma_{y}$ [m]')

    spz = fig. add_subplot(3, 1, 3, sharex=spx)
    spz.plot(np.std(mon.zeta, axis=0))
    spz.axhline(np.sqrt(ez * tw_rad.betz0), color='red')
    spz.set_ylabel(r'$\sigma_{z}$ [m]')

    plt.show()