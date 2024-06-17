sudo cargo flamegraph -- \
--mode build \
--data data/clipped_data.pkl \
-o test.model \
--hidden-layers 24 \
--classes $(cat data/classes.txt) \
-e 10 \
--learning-rate 0.05 \
--validate-each-epoch 