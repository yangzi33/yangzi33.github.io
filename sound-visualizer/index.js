window.onload = function () {
    "use strict";
    var paths = document.getElementsByTagName('path');
    var visualizer = document.getElementById('visualizer');
    var mask = visualizer.getElementById('mask');
    var h = document.getElementsByTagName('h1')[0];
    var hSub = document.getElementsByTagName('h1')[1];
    var AudioContext;
    var audioContent;
    var start = false;
    var permission = false;
    var path;
    var seconds = 0;
    var loud_volume_threshold = 30;
    
    var soundAllowed = function (stream) {
        permission = true;
        var audioStream = audioContent.createMediaStreamSource( stream );
        var analyser = audioContent.createAnalyser();
        var fftSize = 1024;

        analyser.fftSize = fftSize;
        audioStream.connect(analyser);

        var bufferLength = analyser.frequencyBinCount;
        var frequencyArray = new Uint8Array(bufferLength);
        
        visualizer.setAttribute('viewBox', '0 0 255 255');
      
        for (var i = 0 ; i < 255; i++) {
            path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
            path.setAttribute('stroke-dasharray', '4,1');
            mask.appendChild(path);
        }
        var doDraw = function () {
            requestAnimationFrame(doDraw);
            if (start) {
                analyser.getByteFrequencyData(frequencyArray);
                var adjustedLength;
                for (var i = 0 ; i < 255; i++) {
                  	adjustedLength = Math.floor(frequencyArray[i]) - (Math.floor(frequencyArray[i]) % 5);
                    paths[i].setAttribute('d', 'M '+ (i) +',255 l 0,-' + adjustedLength);
                }
            }
            else {
                for (var i = 0 ; i < 255; i++) {
                    paths[i].setAttribute('d', 'M '+ (i) +',255 l 0,-' + 0);
                }
            }
        }
        var showVolume = function () {
            setTimeout(showVolume, 500);
            if (start) {
                analyser.getByteFrequencyData(frequencyArray);
                var total = 0
                for(var i = 0; i < 255; i++) {
                   var x = frequencyArray[i];
                   total += x * x;
                }
                var rms = Math.sqrt(total / bufferLength);
                h.innerHTML = "Speak";
            } else {
                h.innerHTML = "Stopped";
                hSub.innerHTML = "";
            }
        }

        doDraw();
        showVolume();
    }

    var soundNotAllowed = function (error) {
        h.innerHTML = "Microphone permission is required.";
        console.log(error);
    }


    document.getElementById('start-button').onclick = function () {
        if (start) {
            start = false;
            this.innerHTML = "<span class='fa fa-play'></span>Start";
            this.className = "start-button";
        }
        else {
            if (!permission) {
                navigator.mediaDevices.getUserMedia({audio:true})
                    .then(soundAllowed)
                    .catch(soundNotAllowed);

                AudioContext = window.AudioContext || window.webkitAudioContext;
                audioContent = new AudioContext();
            }
            start = true;
            this.innerHTML = "<span class='fa fa-stop'></span>Stop";
            this.className = "stop-button";
        }
    };
};