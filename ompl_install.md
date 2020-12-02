1. Install dependencies with `sudo apt-get -y install g++ cmake pkg-config libboost-serialization-dev libboost-filesystem-dev libboost-system-dev libboost-program-options-dev libboost-test-dev libeigen3-dev libode-dev wget libyaml-cpp-dev`
2. Install Python dependencies:
    - source your MuJoCo virtualenv's `bin/activate`
    - `pip3 install -vU https://github.com/CastXML/pygccxml/archive/develop.zip pyplusplus`
    - If Ubuntu >= 19.10, `sudo apt-get install castxml`; else:
        - `wget -q -O- https://data.kitware.com/api/v1/file/5e8b740d2660cbefba944189/download | tar zxf - -C ${HOME}/installations`
        - add the following to your .bashrc: `export PATH=${HOME}/installations/castxml/bin:${PATH}`, then source .bashrc and re-source virtualenv
    - `sudo apt-get install libboost-python-dev`
    - If Ubuntu >= 17.10, `sudo apt-get -y install libboost-numpy-dev python3-numpy`
    - If Ubuntu >= 19.10, `sudo apt-get -y install pypy3`
3. `git clone https://github.com/ompl/ompl.git`
4. `cd ompl/ && git checkout 1.5.1`
5. `mkdir build && cd build/`
6. `export venv-path=</path/to/your/venv> && export venv-python=<venv python version>` e.g. `export venv-path=$HOME/installations/virtualenvs/mujoco && export venv-python=3.6`
6. `cmake -DCMAKE_INSTALL_PREFIX=${HOME}/installations -DCASTXML=${HOME}/installations/castxml/bin/castxml -DOMPL_PYTHON_INSTALL_DIR=${venv-path}/lib/python${venv-python}/site-packages/ -DPYTHON_EXEC=${venv-path}/bin/python -DPYTHON_INCLUDE_DIRS=/usr/include/python${venv-python}m -DPYTHON_LIBRARIES=/usr/lib/x86_64-linux-gnu/libpython${venv-python}m.so -DPY_PYGCCXML=${venv-path}/lib/python${venv-python}/site-packages/pygccxml -DPY_PYPLUSPLUS=${venv-path}/lib/python${venv-python}/site-packages/pyplusplus ..`
    - Note that the DPYTHON_INCLUDE_DIRS argument was originally `-DPYTHON_INCLUDE_DIRS=/home/abba/msu_ws/msu-env/include/python3.5m`. The corresponding python3.6m directory was empty for me, so I replaced it with my global `/usr/include/python3.6m`.
7. `make update_bindings`
8. `make -j8`
9. `make install`