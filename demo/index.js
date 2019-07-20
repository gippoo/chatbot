let encoder;
let decoder;

async function load_models() {
    console.log("Loading...");

    encoder = await tf.loadLayersModel('https://gippoo.github.io/chatbot/encoder2/model.json');
    console.log("Encoder loaded");
    
    decoder = await tf.loadLayersModel('https://gippoo.github.io/chatbot/decoder2/model.json');
    console.log("Decoder loaded");
	
    tf.tidy(() => {
        let warmUpEnc = tf.zeros([1, 89, 52]);
        let warmUpStates = encoder.predict(warmUpEnc);
        let warmUpDec = [tf.zeros([1, 1]), warmUpStates[0], warmUpStates[1]];
        decoder.predict(warmUpDec);
    });

    console.log("Warmed Up");
}
	
load_models();

var gifs = ['https://thumbs.gfycat.com/EnormousIdleChickadee.webp', 'https://thumbs.gfycat.com/BogusGloriousCuscus.webp', 
	    'https://thumbs.gfycat.com/DapperBelatedGoitered.webp'];
	    
function updateScroll(){
    var element = document.getElementById("text");
    element.scrollTop = element.scrollHeight;
}
 
function change_div() {
    let prompt = document.getElementById("prompt").value;
    prompt = prompt.toLowerCase();

    document.getElementById("text").innerHTML += 'You: '+prompt+'<br>';
	
    let prompt_vec = [];
    
    for (let i=0; i<prompt.length; i++) {
    	prompt_vec.push(prompt_char_to_ix[prompt[i]]);
    }
    while (prompt_vec.length < 89) {
    	prompt_vec.push(52);
    }
    
    let one_hot = tf.oneHot(tf.tensor1d(prompt_vec, 'int32'), 52);

    let input = one_hot.reshape([1,89,52]);
    
    let states_value = encoder.predict(input);
    
    let target_seq = tf.tensor2d([reply_char_to_ix['sos']], [1, 1]);
	
    let stop_condition = false;
    let decoded_sentence = '';
    let decoder_input = [target_seq, states_value[0], states_value[1]];


    while (!stop_condition) {
    	let pred = decoder.predict(decoder_input);
    	let output_tokens = pred[0];
    	let h = pred[1];
    	let c = pred[2];
	
    	let sampled_token_index = parseInt(output_tokens.squeeze().argMax().dataSync());
    	let sampled_char = reply_ix_to_char[sampled_token_index];
        decoded_sentence += sampled_char;
	    
    	if (sampled_char == 'eos' || decoded_sentence.length > 92) {
            stop_condition = true;
    	}
	
        target_seq = tf.tensor2d([sampled_token_index], [1, 1]);
        
        states_value = [h, c];
        decoder_input = [target_seq, states_value[0], states_value[1]];
    }
	
    document.getElementById("text").innerHTML += 'MayaBot: '+decoded_sentence+'<br><br>';
    document.getElementById("prompt").value = '';
    document.getElementById("image").src = gifs[Math.floor(Math.random()*gifs.length)];
    updateScroll();
}

const node = document.getElementById("prompt");
node.addEventListener("keyup", function(event) {
    if (event.key === "Enter") {
    	tf.tidy(() => {
            change_div();
    	});
    	console.log(tf.memory());
    }
});
