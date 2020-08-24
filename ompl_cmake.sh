
cmake -DCMAKE_INSTALL_PREFIX=$HOME/installations \
      -DCASTXML=${HOME}/installations/castxml/bin/castxml \
      -DOMPL_PYTHON_INSTALL_DIR=/home/abba/msu_ws/msu-env/lib/python3.5/site-packages/ \
      -DPYTHON_EXEC=/home/abba/msu_ws/msu-env/bin/python \
      -DPYTHON_INCLUDE_DIRS=/home/abba/msu_ws/msu-env/include/python3.5m \
      -DPYTHON_LIBRARIES=/usr/lib/x86_64-linux-gnu/libpython3.5m.so \
      -DPY_PYGCCXML=/home/abba/msu_ws/msu-env/lib/python3.5/site-packages/pygccxml \
      -DPY_PYPLUSPLUS=/home/abba/msu_ws/msu-env/lib/python3.5/site-packages/pyplusplus \
      ..
