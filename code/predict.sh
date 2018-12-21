"Total argument: $#"
echo "Script name: $0"
echo "Argument 1: $1"#ta_input.txt
echo "Argument 2: $2"#ta_output.txt
length_vocab_name="output_lang.p"
length_noatten_encoder_name="44_1201_encoder1_length.pt"
length_noatten_decoder_name="44_1201_decoder1_length.pt"

wget "https://drive.google.com/uc?export=download&id=145h4AYWkd2WnF4mvIagLk9Dditp6X-y8" -O $length_vocab_name
wget "https://drive.google.com/uc?export=download&id=1dhouxB0YPNvI80qwegmwsQKZ_7IeIoIH" -O $length_noatten_encoder_name
wget "https://drive.google.com/uc?export=download&id=1HYgS5-jvC-SMCopU4FawT6CbDaYrjmcT" -O $length_noatten_decoder_name
python3 prediction.py $1 $2 $length_vocab_name $length_noatten_encoder_name $length_noatten_decoder_name
