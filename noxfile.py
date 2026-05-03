import nox

nox.options.sessions = ("test",)
locations = "noxfile.py"


@nox.session
def test(session):
    session.install("pytest", "pytest-cov")
    session.install("-e", ".")

    session.run("pytest", "--cov=metaheuristic_designer", "--cov-report=term-missing", "--cov-report=html", "--continue-on-collection-errors")
