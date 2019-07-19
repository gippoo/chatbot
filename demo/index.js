async function init() {
  encoder = await tf.loadLayersModel('https://gippoo.github.io/chatbot/encoder/model.json');
	decoder = await tf.loadLayersModel('https://gippoo.github.io/chatbot/decoder/model.json');
  
  decoder.predict([tf.zeros([1,1,51]), encoder.predict(tf.zeros([1,89,52]))[0], encoder.predict(tf.zeros([1,89,52]))[1]]);
}
	
init();
	
var prompt_char_to_ix = {'y': 0,
 '[': 1,
 '2': 2,
 'v': 3,
 'h': 4,
 'r': 5,
 ')': 6,
 "'": 7,
 'j': 8,
 '1': 9,
 '3': 10,
 ',': 11,
 '4': 12,
 'n': 13,
 'm': 14,
 '8': 15,
 'w': 16,
 '(': 17,
 'g': 18,
 '7': 19,
 '#': 20,
 'a': 21,
 '.': 22,
 ':': 23,
 'x': 24,
 'e': 25,
 'u': 26,
 '5': 27,
 'q': 28,
 'b': 29,
 '&': 30,
 'f': 31,
 '!': 32,
 'c': 33,
 'i': 34,
 '0': 35,
 't': 36,
 'l': 37,
 '"': 38,
 '-': 39,
 'p': 40,
 '6': 41,
 '$': 42,
 'z': 43,
 '9': 44,
 ']': 45,
 '?': 46,
 ' ': 47,
 'd': 48,
 's': 49,
 'k': 50,
 'o': 51}
 
 var prompt_ix_to_char = {0: 'y',
 1: '[',
 2: '2',
 3: 'v',
 4: 'h',
 5: 'r',
 6: ')',
 7: "'",
 8: 'j',
 9: '1',
 10: '3',
 11: ',',
 12: '4',
 13: 'n',
 14: 'm',
 15: '8',
 16: 'w',
 17: '(',
 18: 'g',
 19: '7',
 20: '#',
 21: 'a',
 22: '.',
 23: ':',
 24: 'x',
 25: 'e',
 26: 'u',
 27: '5',
 28: 'q',
 29: 'b',
 30: '&',
 31: 'f',
 32: '!',
 33: 'c',
 34: 'i',
 35: '0',
 36: 't',
 37: 'l',
 38: '"',
 39: '-',
 40: 'p',
 41: '6',
 42: '$',
 43: 'z',
 44: '9',
 45: ']',
 46: '?',
 47: ' ',
 48: 'd',
 49: 's',
 50: 'k',
 51: 'o'}
 
 var reply_char_to_ix = {'y': 0,
 '[': 1,
 '2': 2,
 'v': 3,
 'h': 4,
 'r': 5,
 ')': 6,
 "'": 7,
 'j': 8,
 '1': 9,
 '\t': 10,
 ',': 11,
 '4': 12,
 'n': 13,
 'm': 14,
 '8': 15,
 'w': 16,
 '7': 17,
 'g': 18,
 '(': 19,
 'a': 20,
 '.': 21,
 ':': 22,
 'x': 23,
 'e': 24,
 'u': 25,
 '5': 26,
 'q': 27,
 'b': 28,
 '&': 29,
 'f': 30,
 '!': 31,
 'c': 32,
 'i': 33,
 '0': 34,
 't': 35,
 'l': 36,
 'p': 37,
 '-': 38,
 '"': 39,
 '6': 40,
 'z': 41,
 '9': 42,
 ']': 43,
 '?': 44,
 ' ': 45,
 'd': 46,
 's': 47,
 '\n': 48,
 'k': 49,
 'o': 50}
 
 var reply_ix_to_char = {0: 'y',
 1: '[',
 2: '2',
 3: 'v',
 4: 'h',
 5: 'r',
 6: ')',
 7: "'",
 8: 'j',
 9: '1',
 10: '\t',
 11: ',',
 12: '4',
 13: 'n',
 14: 'm',
 15: '8',
 16: 'w',
 17: '7',
 18: 'g',
 19: '(',
 20: 'a',
 21: '.',
 22: ':',
 23: 'x',
 24: 'e',
 25: 'u',
 26: '5',
 27: 'q',
 28: 'b',
 29: '&',
 30: 'f',
 31: '!',
 32: 'c',
 33: 'i',
 34: '0',
 35: 't',
 36: 'l',
 37: 'p',
 38: '-',
 39: '"',
 40: '6',
 41: 'z',
 42: '9',
 43: ']',
 44: '?',
 45: ' ',
 46: 'd',
 47: 's',
 48: '\n',
 49: 'k',
 50: 'o'}
 
var gifs = ['https://thumbs.gfycat.com/EnormousIdleChickadee.webp', 'https://thumbs.gfycat.com/BogusGloriousCuscus.webp', 
	    'https://thumbs.gfycat.com/DapperBelatedGoitered.webp'];

function updateScroll(){
    var element = document.getElementById("text");
    element.scrollTop = element.scrollHeight;
}
 
async function change_div() {
    let prompt = document.getElementById("prompt").value;
    prompt = prompt.toLowerCase();

    document.getElementById("text").innerHTML += 'You: '+prompt+'<br>';
	
    let prompt_vec = [];
    
    for (let i=0; i<prompt.length; i++) {
    	prompt_vec.push(prompt_char_to_ix[prompt[i]])
    }
    while (prompt_vec.length < 89) {
    	prompt_vec.push(52);
    }
    
    let one_hot = tf.oneHot(tf.tensor1d(prompt_vec, 'int32'), 52);

    let input = one_hot.reshape([1,89,52]);
    
    let states_value = encoder.predict(input);
    
    let target_seq = [];
    for (let i=0;i<51;i++) {
	if (i != reply_char_to_ix['\t']) {
		target_seq.push(0);	
	} else {
		target_seq.push(1);
	}
    }
	
    target_seq = tf.tensor3d(target_seq, [1, 1, 51])
	
    let stop_condition = false;
    let decoded_sentence = '';
    let decoder_input = [target_seq, states_value[0], states_value[1]];


    while (!stop_condition) {
    	let pred = decoder.predict(decoder_input);
	let output_tokens = pred[0];
	let h = pred[1];
	let c = pred[2];
	
	let sampled_token_index = await parseInt(output_tokens.squeeze().argMax().data());
	let sampled_char = reply_ix_to_char[sampled_token_index]
        decoded_sentence += sampled_char
	    
	if (sampled_char == '\n' || decoded_sentence.length > 92) {
            stop_condition = true
	}
	
	 target_seq = [];
	 for (let i=0;i<51;i++) {
		if (i != reply_char_to_ix[sampled_char]) {
			target_seq.push(0);	
		} else {
			target_seq.push(1);
		}
	  }
	  target_seq = tf.tensor3d(target_seq, [1, 1, 51])
        
          states_value = [h, c]
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
