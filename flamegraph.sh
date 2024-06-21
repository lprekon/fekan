sudo cargo flamegraph -- \
--data data/clipped_data.pkl \
build classifier \
-o test.model \
--hidden-layer-sizes 24 \
--classes $(cat data/classes.txt) \
-e 10 \
--learning-rate 0.05 \
--validate-each-epoch 