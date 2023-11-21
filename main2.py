<!doctype html>
<html>

<head>
  <title>Speech Recognition Demo</title>
  <meta charset="utf-8" />
  <meta http-equiv="Content-type" content="text/html; charset=utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    *,
    *::before,
    *::after {
      box-sizing: border-box;
    }

    body,
    main {
      margin: 0;
      padding: 0;
      min-width: 100%;
      min-height: 100vh;
      font-family: sans-serif;
      text-align: center;
      color: #fff;
      background: #000;
    }

    button {
      position: absolute;
      left: 50%;
      top: 50%;
      width: 5em;
      height: 2em;
      margin-left: -2.5em;
      margin-top: -1em;
      z-index: 100;
      padding: .25em .5em;
      color: #fff;
      background: #000;
      border: 1px solid #fff;
      border-radius: 4px;
      cursor: pointer;
      font-size: 1.15em;
      font-weight: 200;
      box-shadow: 0 0 10px rgba(255, 255, 255, 0.5);
      transition: box-shadow .5s;
    }

    button:hover {
      box-shadow: 0 0 30px 5px rgba(255, 255, 255, 0.75);
    }

    main {
      position: relative;
      display: flex;
      justify-content: center;
      align-items: center;
    }

    main>div {
      display: inline-block;
      width: 3px;
      height: 100px;
      margin: 0 7px;
      background: currentColor;
      transform: scaleY(.5);
      opacity: .25;
    }

    main.error {
      color: #f7451d;
      min-width: 20em;
      max-width: 30em;
      margin: 0 auto;
      white-space: pre-line;
    }

    #transcript {
      position: fixed;
      top: 60%;
      width: 100vw;
      padding-left: 20vw;
      padding-right: 20vw;
      font-size: x-large;
    }
  </style>
</head>

<body>
  <main>
    <button onclick="init()">start</button>
  </main>
  <div id="transcript"></div>
</body>


<script>
  class AudioVisualizer {
    constructor(audioContext, processFrame, processError) {
      this.audioContext = audioContext;
      this.processFrame = processFrame;
      this.connectStream = this.connectStream.bind(this);
      navigator.mediaDevices.getUserMedia({ audio: true, video: false })
        .then(this.connectStream)
        .catch((error) => {
          if (processError) {
            processError(error);
          }
        });
    }

    connectStream(stream) {
      this.analyser = this.audioContext.createAnalyser();
      const source = this.audioContext.createMediaStreamSource(stream);
      console.log(source)
      source.connect(this.analyser);
      this.analyser.smoothingTimeConstant = 0.5;
      this.analyser.fftSize = 32;

      this.initRenderLoop(this.analyser);
    }

    initRenderLoop() {
      const frequencyData = new Uint8Array(this.analyser.frequencyBinCount);
      const processFrame = this.processFrame || (() => { });

      const renderFrame = () => {
        this.analyser.getByteFrequencyData(frequencyData);
        processFrame(frequencyData);

        requestAnimationFrame(renderFrame);
      };
      requestAnimationFrame(renderFrame);
    }
  }

  const visualMainElement = document.querySelector('main');
  const visualValueCount = 16;
  let visualElements;
  const createDOMElements = () => {
    let i;
    for (i = 0; i < visualValueCount; ++i) {
      const elm = document.createElement('div');
      visualMainElement.appendChild(elm);
    }

    visualElements = document.querySelectorAll('main div');
  };
  createDOMElements();

  const startTranscriptions = () => {
    fetch("/start_asr").then(res => res.json()).then(data => console.log(data))
    setInterval(() => {
      fetch("/get_audio").then(res => res.json()).then(data => {
        let doc = document.getElementById("transcript")
        if (data !== "") {
          doc.innerHTML = data
        }
      })
    }, 100)
  };

  const init = () => {
    // Creating initial DOM elements
    const audioContext = new AudioContext();
    const initDOM = () => {
      visualMainElement.innerHTML = '';
      createDOMElements();
    };
    initDOM();

    // Swapping values around for a better visual effect
    const dataMap = { 0: 15, 1: 10, 2: 8, 3: 9, 4: 6, 5: 5, 6: 2, 7: 1, 8: 0, 9: 4, 10: 3, 11: 7, 12: 11, 13: 12, 14: 13, 15: 14 };
    const processFrame = (data) => {
      const values = Object.values(data);
      let i;
      for (i = 0; i < visualValueCount; ++i) {
        const value = values[dataMap[i]] / 255;
        const elmStyles = visualElements[i].style;
        elmStyles.transform = `scaleY( ${value} )`;
        elmStyles.opacity = Math.max(.25, value);
      }
    };

    const processError = () => {
      visualMainElement.classList.add('error');
      visualMainElement.innerText = 'Please allow access to your microphone in order to see this demo.\nNothing bad is going to happen... hopefully :P';
    }

    const a = new AudioVisualizer(audioContext, processFrame, processError);

    startTranscriptions()
  };
</script>

</html>
