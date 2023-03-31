import nox
import nox_poetry


nox.options.sessions = "tests",
locations = "src", "noxfile.py"

@nox_poetry.session
def tests(session):
    session.install('.')
    session.install('pytest')
    session.run('pytest')