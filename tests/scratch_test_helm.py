import subprocess


def test_helm_binary_downloads(helm_binary_path):
    print(subprocess.run([helm_binary_path, "version"], capture_output=True, text=True).stdout)
