clean:
	pip uninstall -y xops
	rm -rf build/ dist/ *.egg-info
	find . -name "*.so" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +

install:
	MAX_JOBS=2 pip install -e . -v --no-build-isolation

re: clean install

check_dbg:
	objdump -h backend/xops.*.so | grep debug