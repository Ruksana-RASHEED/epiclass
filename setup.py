from setuptools import setup

setup(
    name='epiclass',
    version='1.0.1',
    packages=['epiclass'],
    url='https://github.com/moink/epiclass',
    license='MIT',
    author='moink',
    author_email='',
    description='Visualization and prediction of epileptic seizure data set',
    install_requires=['keras', 'pandas', 'joblib', 'matplotlib', 'seaborn',
                      'scikit-learn', 'flask>=1.0.2', 'flask_restful',
                      'TensorFlow', 'numpy'],
    scripts=['run_epiclass.py', 'api.py', 'test_deployment.py']
)