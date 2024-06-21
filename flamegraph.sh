sudo cargo flamegraph -- \
--data data/clipped_data.pkl \
build classifier \
--no-save \
--hidden-layer-sizes 24 \
--classes $(cat data/classes.txt) \
-e 2 \
--learning-rate 0.05 \
--validate-each-epoch 