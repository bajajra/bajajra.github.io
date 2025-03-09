// Dynamically update the footer year
const yearSpan = document.getElementById("year");
if (yearSpan) {
  yearSpan.textContent = new Date().getFullYear();
}

// Smoothly scroll to Profile section on Home page
function scrollToProfile() {
  const profileSection = document.getElementById("profileSection");
  if (profileSection) {
    profileSection.scrollIntoView({ behavior: "smooth" });
  }
}

// Navbar toggle for mobile
const navToggle = document.getElementById("navToggle");
const navLinks = document.getElementById("navLinks");
if (navToggle && navLinks) {
  navToggle.addEventListener("click", () => {
    navLinks.classList.toggle("show");
  });
}

// Toggle review details
document.addEventListener("click", (e) => {
  if (e.target && e.target.classList.contains("toggle-details")) {
    const details = e.target.nextElementSibling;
    if (details.classList.contains("hidden")) {
      details.classList.remove("hidden");
      e.target.textContent = "Hide Full Review";
    } else {
      details.classList.add("hidden");
      e.target.textContent = "View Full Review";
    }
  }
});
