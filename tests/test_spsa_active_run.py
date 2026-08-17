"""Regression tests for workers retaining claims from superseded SPSA runs."""

from web.app import create_app
from web.database import db
from web.models import SpsaIteration, SpsaRun
from compete.spsa.master import _activate_run


def _run(name, active):
    run = SpsaRun(name=name, is_active=active)
    db.session.add(run)
    db.session.flush()
    return run


def _iteration(run, number, status='pending'):
    iteration = SpsaIteration(
        run_id=run.id,
        iteration_number=number,
        timelow_ms=100,
        timehigh_ms=200,
        target_games=10,
        ref_target_games=10,
        status=status,
    )
    db.session.add(iteration)
    db.session.flush()
    return iteration


def _client(monkeypatch, tmp_path):
    monkeypatch.setenv('DATABASE_URL', f"sqlite:///{tmp_path / 'spsa.db'}")
    monkeypatch.setenv('SPSA_WORKER_API_KEY', 'test-key')
    app = create_app()
    app.config['TESTING'] = True
    with app.app_context():
        db.create_all()
    return app.test_client(), app


def test_work_prefers_newest_active_run(monkeypatch, tmp_path):
    client, app = _client(monkeypatch, tmp_path)
    with app.app_context():
        old_run = _run('old', True)
        _iteration(old_run, 1)
        new_run = _run('new', True)
        expected = _iteration(new_run, 1)
        db.session.commit()
        expected_id = expected.id

    response = client.get('/api/spsa/work', headers={'X-API-Key': 'test-key'})

    assert response.status_code == 200
    assert response.get_json()['id'] == expected_id


def test_inactive_run_results_are_ignored(monkeypatch, tmp_path):
    client, app = _client(monkeypatch, tmp_path)
    with app.app_context():
        inactive = _run('inactive', False)
        iteration = _iteration(inactive, 1, status='in_progress')
        db.session.commit()
        iteration_id = iteration.id

    response = client.post(
        f'/api/spsa/iterations/{iteration_id}/results',
        headers={'X-API-Key': 'test-key'},
        json={'games': 2, 'plus_wins': 1, 'minus_wins': 1, 'draws': 0},
    )

    assert response.status_code == 200
    assert response.get_json() == {
        'status': 'ignored', 'accepted': False,
        'reason': 'iteration is not active', 'remaining': 0,
    }
    with app.app_context():
        assert db.session.get(SpsaIteration, iteration_id).games_played == 0


def test_inactive_run_reference_results_are_ignored(monkeypatch, tmp_path):
    client, app = _client(monkeypatch, tmp_path)
    with app.app_context():
        inactive = _run('inactive', False)
        iteration = _iteration(inactive, 1, status='ref_pending')
        db.session.commit()
        iteration_id = iteration.id

    response = client.post(
        f'/api/spsa/iterations/{iteration_id}/ref-results',
        headers={'X-API-Key': 'test-key'},
        json={'games': 2, 'wins': 1, 'losses': 1, 'draws': 0},
    )

    assert response.status_code == 200
    assert response.get_json()['status'] == 'ignored'
    with app.app_context():
        assert db.session.get(SpsaIteration, iteration_id).ref_games_played == 0


def test_activation_abandons_only_outgoing_run_work(monkeypatch, tmp_path):
    _, app = _client(monkeypatch, tmp_path)
    with app.app_context():
        outgoing = _run('outgoing', True)
        outgoing_iteration = _iteration(outgoing, 1, status='in_progress')
        incoming = _run('incoming', False)
        incoming_iteration = _iteration(incoming, 1, status='in_progress')
        db.session.commit()
        outgoing_id = outgoing_iteration.id
        incoming_id = incoming_iteration.id

        _activate_run(db, SpsaRun, incoming)
        assert db.session.get(SpsaIteration, outgoing_id).status == 'abandoned'
        assert db.session.get(SpsaIteration, incoming_id).status == 'in_progress'

        # Restarting the master for its already-active run must resume work.
        _activate_run(db, SpsaRun, incoming)
        assert db.session.get(SpsaIteration, incoming_id).status == 'in_progress'
