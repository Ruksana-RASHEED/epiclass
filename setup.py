from setuptools import setup

setup(
    name='epiclass',
    version='1.0.0',
    packages=['epiclass'],
    url='https://github.com/moink/epiclass',
    license='MIT',
    author='moink',
    author_email='',
    description='Visualization and prediction of epileptic seizure data set',
    install_requires=['keras', 'pandas', 'joblib', 'matplotlib', 'seaborn',
                      'scikit-learn', 'flask>=1.0.2', 'flask_restful',
                      'tensorflow'],
    scripts=['run_epiclass', 'api', 'test_deployment']
)