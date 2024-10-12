document.addEventListener("DOMContentLoaded", () => {
	const sidebar = document.querySelector(".sphinxsidebar");

	sidebar.addEventListener("click", (event) => {
		if (event.target === sidebar) {
			sidebar.classList.toggle("opened");
		}
	});
});
