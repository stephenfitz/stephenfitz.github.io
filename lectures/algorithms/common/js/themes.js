Reveal.on('ready', () => {
    const LIGHT_THEME = "../common/reveal/dist/theme/white.css";
    const DARK_THEME = "../common/reveal/dist/theme/black.css";
    const THEME_LINK_ID = "theme";

    const themeLink = document.getElementById(THEME_LINK_ID);
    if (!themeLink) {
        console.warn("Theme link with id='theme' not found.");
        return;
    }

    // Restore theme from localStorage if available
    const savedTheme = localStorage.getItem("reveal-theme");
    if (savedTheme && savedTheme !== themeLink.href) {
        themeLink.href = savedTheme;
    }

    function getCurrentTheme() {
        return themeLink.href.includes("white.css") ? "light" : "dark";
    }

    function switchTheme() {
        const newTheme = getCurrentTheme() === "light" ? DARK_THEME : LIGHT_THEME;
        themeLink.href = newTheme;
        localStorage.setItem("reveal-theme", newTheme);
        document.body.setAttribute("data-theme", newTheme.includes("white.css") ? "light" : "dark");
        if (newTheme === LIGHT_THEME) {
            document.getElementById("highlight-theme").href = "../common/reveal/plugin/highlight/github.css"
        } else {
            document.getElementById("highlight-theme").href = "../common/reveal/plugin/highlight/nord.min.css"
        }
    }

    function lightTheme() {
        const newTheme = LIGHT_THEME;
        themeLink.href = newTheme;
        localStorage.setItem("reveal-theme", newTheme);
        document.body.setAttribute("data-theme", newTheme.includes("white.css") ? "light" : "dark");
        document.getElementById("highlight-theme").href = "../common/reveal/plugin/highlight/github.css"
    }

    function darkTheme() {
        const newTheme = DARK_THEME;
        themeLink.href = newTheme;
        localStorage.setItem("reveal-theme", newTheme);
        document.body.setAttribute("data-theme", newTheme.includes("white.css") ? "light" : "dark");
        document.getElementById("highlight-theme").href = "../common/reveal/plugin/highlight/nord.min.css"
    }

    // Keyboard listener: press 't' to toggle theme
    document.addEventListener("keydown", (event) => {
        if (event.key.toLowerCase() === "t") {
        switchTheme();
        }
    });
    darkTheme();
    // lightTheme();
    Object.assign(window, { switchTheme, lightTheme, darkTheme });
});
