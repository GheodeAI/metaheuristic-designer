import nox

nox.options.sessions = ("test",)
locations = "noxfile.py"


@nox.session
def test(session):
    session.install("-e", ".[test]")

    session.run("pytest", "--cov=metaheuristic_designer", "--cov-report=term-missing", "--cov-report=html", "--continue-on-collection-errors")
