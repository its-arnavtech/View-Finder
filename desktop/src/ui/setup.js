"use strict";
(function () {
  var setup = window.viewfinderSetup;
  var button = document.getElementById("start");
  var status = document.getElementById("status");
  var log = document.getElementById("log");
  var MAX_LOG = 200000;

  setup.onLog(function (text) {
    log.hidden = false;
    log.textContent = (log.textContent + text).slice(-MAX_LOG);
    log.scrollTop = log.scrollHeight;
  });

  button.addEventListener("click", function () {
    button.disabled = true;
    status.className = "";
    status.textContent = "Setting up… (keep the app open)";
    setup.start().catch(function (error) {
      button.disabled = false;
      button.textContent = "Try again";
      status.className = "error";
      status.textContent = error.message;
    });
  });
})();
