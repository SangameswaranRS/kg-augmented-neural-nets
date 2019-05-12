# Bash Script to start the stanford core NLP server
echo "[INFO] Starting Server"
cd /home/sangameswaran/stanford-core-nlp/
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
