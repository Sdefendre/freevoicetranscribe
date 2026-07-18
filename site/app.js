(() => {
  "use strict";

  const demo = document.querySelector("[data-demo]");
  const demoText = document.querySelector("[data-demo-text]");
  const demoStatus = document.querySelector("[data-demo-status]");
  const demoDetail = document.querySelector("[data-demo-detail]");
  const replayButton = document.querySelector("[data-demo-replay]");
  const motionPreference = window.matchMedia("(prefers-reduced-motion: reduce)");
  const skipLink = document.querySelector(".skip-link");
  const mainContent = document.querySelector("#main");

  const finalText = "a clear thought, placed right where you left the cursor.";
  const stateCopy = {
    recording: ["Listening", "Release fn to insert. Press esc to cancel."],
    transcribing: ["Transcribing", "Processing locally on this Mac."],
    inserting: ["Inserting text", "Returning to your app."],
    complete: ["Ready", "Hold fn and speak. Release to insert."],
  };

  let demoVisible = false;
  let demoRunning = false;
  let scheduledTasks = [];

  const setState = (state) => {
    const [status, detail] = stateCopy[state];
    demo.dataset.state = state;
    demoStatus.textContent = status;
    demoDetail.textContent = detail;
  };

  const schedule = (callback, delay) => {
    const task = window.setTimeout(callback, delay);
    scheduledTasks.push(task);
  };

  const stopDemo = () => {
    scheduledTasks.forEach((task) => window.clearTimeout(task));
    scheduledTasks = [];
    demoRunning = false;
  };

  const showFinalFrame = () => {
    stopDemo();
    demoText.textContent = finalText;
    setState("complete");
  };

  const typeSentence = () => {
    const characters = Array.from(finalText);
    const typingDuration = 2200;
    const interval = typingDuration / characters.length;

    characters.forEach((character, index) => {
      schedule(() => {
        demoText.textContent += character;
      }, index * interval);
    });
  };

  const playDemo = () => {
    if (motionPreference.matches) {
      showFinalFrame();
      return;
    }

    stopDemo();
    demoRunning = true;
    demoText.textContent = "";
    setState("recording");

    schedule(() => setState("transcribing"), 2800);
    schedule(() => {
      setState("inserting");
      typeSentence();
    }, 4000);
    schedule(() => {
      demoText.textContent = finalText;
      setState("complete");
    }, 6400);
    schedule(() => {
      demoRunning = false;
      if (demoVisible && !document.hidden) {
        playDemo();
      }
    }, 8000);
  };

  const syncMotionPreference = () => {
    if (motionPreference.matches) {
      replayButton.disabled = true;
      replayButton.textContent = "Demo shown without motion";
      showFinalFrame();
      return;
    }

    replayButton.disabled = false;
    replayButton.innerHTML = `
      <svg viewBox="0 0 18 18" aria-hidden="true">
        <path d="M14 6V2m0 0h-4m4 0-2.1 2.1a5.5 5.5 0 1 0 1.3 7.7"></path>
      </svg>
      Replay demo
    `;

    if (demoVisible && !demoRunning && !document.hidden) {
      playDemo();
    }
  };

  replayButton.addEventListener("click", () => {
    if (!motionPreference.matches) {
      playDemo();
    }
  });

  skipLink.addEventListener("click", () => {
    window.requestAnimationFrame(() => {
      mainContent.focus({ preventScroll: true });
    });
  });

  if ("IntersectionObserver" in window) {
    const demoObserver = new IntersectionObserver(
      (entries) => {
        const entry = entries[0];
        demoVisible = entry.isIntersecting && entry.intersectionRatio >= 0.3;

        if (demoVisible && !demoRunning && !document.hidden) {
          playDemo();
        } else if (!demoVisible) {
          stopDemo();
        }
      },
      { threshold: [0, 0.3, 0.6] },
    );

    demoObserver.observe(demo);
  } else {
    demoVisible = true;
    playDemo();
  }

  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      stopDemo();
    } else if (demoVisible) {
      playDemo();
    }
  });

  if (typeof motionPreference.addEventListener === "function") {
    motionPreference.addEventListener("change", syncMotionPreference);
  } else {
    motionPreference.addListener(syncMotionPreference);
  }

  document.querySelectorAll("[data-year]").forEach((year) => {
    year.textContent = String(new Date().getFullYear());
  });

  syncMotionPreference();
})();
