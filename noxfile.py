import nox

nox.options.sessions = "test",
locations = "noxfile.py"

@nox.session
def test(session):
    session.install('.')
    session.install('pytest', 'coverage')
    session.run('coverage', 'run', '--data-file', '.coverage.nox', '--parallel', '-m', 'pytest')
    session.run('coverage', 'html', '--data-file', '.coverage.nox')