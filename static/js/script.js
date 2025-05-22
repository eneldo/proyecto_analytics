document.querySelectorAll(".collapsible").forEach(btn => {
    btn.onclick = () => {
      btn.classList.toggle("active");
      const content = btn.nextElementSibling;
      content.style.display = content.style.display === "block" ? "none" : "block";
    };
  });
  