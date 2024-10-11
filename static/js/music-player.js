$(function () {
    var playerTrack = $("#player-track"),
      bgArtwork = $("#bg-artwork"),
      bgArtworkUrl,
      albumName = $("#album-name"),
      albumArt = $("#album-art"),
      sArea = $("#s-area"),
      seekBar = $("#seek-bar"),
      trackTime = $("#track-time"),
      insTime = $("#ins-time"),
      sHover = $("#s-hover"),
      playPauseButton = $("#play-pause-button"),
      i = playPauseButton.find("i"),
      tProgress = $("#current-time"),
      tTime = $("#track-length"),
      seekT,
      seekLoc,
      seekBarPos,
      cM,
      ctMinutes,
      ctSeconds,
      curMinutes,
      curSeconds,
      durMinutes,
      durSeconds,
      playProgress,
      bTime,
      nTime = 0,
      buffInterval = null,
      tFlag = false,
      albums = [
		// Audio
        [
            "Fake-dogbark",
            "Real-dogbark",
            "Fake-keyboard",
            "Real-keyboard"
      	],
		// Music
        [
            "Fake-music01",
            "Real-music01",
			"Fake-music02",
			"Real-music02"
        ],
		// Speech
        [
            "Fake-speech01",
			"Real-speech01",
			"Fake-speech02",
			"Real-speech02"
        ],
		// Singing
		[
			"Fake-singing01",
			"Real-singing01",
			"Fake-singing02",
			"Real-singing02"
		]
    ]
      albumArtworks = [
		// Audio
		[
			"audio_fake_1",
			"audio_real_1",
			"audio_fake_2",
			"audio_real_2"
		],
		// Music
		[
			"music_fake_1",
			"music_real_1",
			"music_fake_2",
			"music_real_2"
		],
		// Speech
		[
			"speech_fake_1",
			"speech_real_1",
			"speech_fake_2",
			"speech_real_2"
		],
		// Singing
		[
			"singing_fake_1",
			"singing_real_1",
			"singing_fake_2",
			"singing_real_2"
		]
	  ],
      trackUrl = [
		// Audio
		[
			"static/img/datacase/audio/audio_dogbark_10_fake.mp3",
			"static/img/datacase/audio/audio_dogbark_17_real.mp3",
			"static/img/datacase/audio/audio_keyboard_9_fake.mp3",
			"static/img/datacase/audio/audio_keyboard_15_real.mp3"
		],
		// Music
		[
			"static/img/datacase/audio/music_fake_6vRBp8jo40Q.mp3",
			"static/img/datacase/audio/music_real_6vRBp8jo40Q.mp3",
			"static/img/datacase/audio/music_fake_rVdI-aD9pq8.mp3",
			"static/img/datacase/audio/music_real_rVdI-aD9pq8.mp3"
		],
		// Speech
		[
			"static/img/datacase/audio/speech_26_fake.mp3",
			"static/img/datacase/audio/speech_26_real.mp3",
			"static/img/datacase/audio/speech_170_fake.mp3",
			"static/img/datacase/audio/speech_170_real.mp3"
		],
		// Singing
		[
			"static/img/datacase/audio/singing_16_fake.mp3",
			"static/img/datacase/audio/singing_16_real.mp3",
			"static/img/datacase/audio/singing_143_fake.mp3",
			"static/img/datacase/audio/singing_143_real.mp3"
		]
      ],
      playPreviousTrackButton = $("#play-previous"),
      playNextTrackButton = $("#play-next"),
      currIndex = -1;

    function playPause() {
      setTimeout(function () {
        if (audio.paused) {
          playerTrack.addClass("active");
          albumArt.addClass("active");
          checkBuffering();
          i.attr("class", "fas fa-pause");
          audio.play();
        } else {
          playerTrack.removeClass("active");
          albumArt.removeClass("active");
          clearInterval(buffInterval);
          albumArt.removeClass("buffering");
          i.attr("class", "fas fa-play");
          audio.pause();
        }
      }, 300);
    }
  
    function showHover(event) {
      seekBarPos = sArea.offset();
      seekT = event.clientX - seekBarPos.left;
      seekLoc = audio.duration * (seekT / sArea.outerWidth());
  
      sHover.width(seekT);
  
      cM = seekLoc / 60;
  
      ctMinutes = Math.floor(cM);
      ctSeconds = Math.floor(seekLoc - ctMinutes * 60);
  
      if (ctMinutes < 0 || ctSeconds < 0) return;
  
      if (ctMinutes < 0 || ctSeconds < 0) return;
  
      if (ctMinutes < 10) ctMinutes = "0" + ctMinutes;
      if (ctSeconds < 10) ctSeconds = "0" + ctSeconds;
  
      if (isNaN(ctMinutes) || isNaN(ctSeconds)) insTime.text("--:--");
      else insTime.text(ctMinutes + ":" + ctSeconds);
  
      insTime.css({ left: seekT, "margin-left": "-21px" }).fadeIn(0);
    }
  
    function hideHover() {
      sHover.width(0);
      insTime.text("00:00").css({ left: "0px", "margin-left": "0px" }).fadeOut(0);
    }
  
    function playFromClickedPos() {
      audio.currentTime = seekLoc;
      seekBar.width(seekT);
      hideHover();
    }
  
    function updateCurrTime() {
      nTime = new Date();
      nTime = nTime.getTime();
  
      if (!tFlag) {
        tFlag = true;
        trackTime.addClass("active");
      }
  
      curMinutes = Math.floor(audio.currentTime / 60);
      curSeconds = Math.floor(audio.currentTime - curMinutes * 60);
  
      durMinutes = Math.floor(audio.duration / 60);
      durSeconds = Math.floor(audio.duration - durMinutes * 60);
  
      playProgress = (audio.currentTime / audio.duration) * 100;
  
      if (curMinutes < 10) curMinutes = "0" + curMinutes;
      if (curSeconds < 10) curSeconds = "0" + curSeconds;
  
      if (durMinutes < 10) durMinutes = "0" + durMinutes;
      if (durSeconds < 10) durSeconds = "0" + durSeconds;
  
      if (isNaN(curMinutes) || isNaN(curSeconds)) tProgress.text("00:00");
      else tProgress.text(curMinutes + ":" + curSeconds);
  
      if (isNaN(durMinutes) || isNaN(durSeconds)) tTime.text("00:00");
      else tTime.text(durMinutes + ":" + durSeconds);
  
      if (
        isNaN(curMinutes) ||
        isNaN(curSeconds) ||
        isNaN(durMinutes) ||
        isNaN(durSeconds)
      )
        trackTime.removeClass("active");
      else trackTime.addClass("active");
  
      seekBar.width(playProgress + "%");
  
      if (playProgress == 100) {
        i.attr("class", "fa fa-play");
        seekBar.width(0);
        tProgress.text("00:00");
        albumArt.removeClass("buffering").removeClass("active");
        clearInterval(buffInterval);
      }
    }
  
    function checkBuffering() {
      clearInterval(buffInterval);
      buffInterval = setInterval(function () {
        if (nTime == 0 || bTime - nTime > 1000) albumArt.addClass("buffering");
        else albumArt.removeClass("buffering");
  
        bTime = new Date();
        bTime = bTime.getTime();
      }, 100);
    }
  
    function selectTrack(type, flag) {
      if (flag == 0 || flag == 1) ++currIndex;
      else --currIndex;
  
      if (currIndex > -1 && currIndex < albumArtworks[type].length) {
        if (flag == 0) i.attr("class", "fa fa-play");
        else {
          albumArt.removeClass("buffering");
          i.attr("class", "fa fa-pause");
        }
		audio.pause();
        seekBar.width(0);
        trackTime.removeClass("active");
        tProgress.text("00:00");
        tTime.text("00:00");
  
        currAlbum = albums[type][currIndex];
        currArtwork = albumArtworks[type][currIndex];
		
        audio.src = trackUrl[type][currIndex];

		// console.log(currAlbum, currArtwork, audio.src);
  
        nTime = 0;
        bTime = new Date();
        bTime = bTime.getTime();
  
        if (flag != 0) {
          audio.play();
          playerTrack.addClass("active");
          albumArt.addClass("active");
  
          clearInterval(buffInterval);
          checkBuffering();
        }
  
        albumName.text(currAlbum);
        albumArt.find("img.active").removeClass("active");
        $("#" + currArtwork).addClass("active");
  
        bgArtworkUrl = $("#" + currArtwork).attr("src");
      } else {
        if (flag == 0 || flag == 1) --currIndex;
        else ++currIndex;
      }
    }
  
    window.initPlayer = function(type) {
      audio = new Audio();
	  // init currindex to -1
	  currIndex = -1;
      selectTrack(type, 0);
  
      audio.loop = false;
  
      playPauseButton.on("click", playPause);
  
      sArea.mousemove(function (event) {
        showHover(event);
      });
  
      sArea.mouseout(hideHover);
  
      sArea.on("click", playFromClickedPos);
  
      $(audio).on("timeupdate", updateCurrTime);
  
      playPreviousTrackButton.on("click", function () {
        selectTrack(type, -1);
      });
      playNextTrackButton.on("click", function () {
        selectTrack(type, 1);
      });
    }
	// Change the audio type
	window.changeAudio = function(type) {
    var buttonElement = document.querySelectorAll('#img-audio-container div button');
    var types = ['audio', 'music', 'speech', 'singing'];
    
	let num = 0;
    // Update the button status
    buttonElement.forEach((button, index) => {
      if (types[index] === type) {
        button.classList.add("active-button");
		num = index;
      } else {
        button.classList.remove("active-button");
      }
    });
    // Select the track based on the type
	currIndex = -1;
	playPreviousTrackButton.off("click").on("click", function () {
		selectTrack(num, -1);
	  });
	  playNextTrackButton.off("click").on("click", function () {
		selectTrack(num, 1);
	  });
    selectTrack(num, 0);
  }
    initPlayer(0);
  });