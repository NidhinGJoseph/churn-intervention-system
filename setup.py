from setuptools import setup, find_packages

setup(
    name="churn-intervention-system",
    version="1.0.0",
    author="Nidhin George Joseph",
    description="Budget-constrained churn intervention system using ML and optimization",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "streamlit",
        "pandas",
        "numpy",
        "scikit-learn",
        "xgboost",
        "matplotlib",
        "seaborn",
        "joblib",
        "plotly"
    ],
    python_requires=">=3.10",
)