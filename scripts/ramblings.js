document.addEventListener("DOMContentLoaded", function () {
    const articles = [
        {
            title: "Building Git from Scratch in Python",
            excerpt: "Everyone uses Git, but do you actually know how it works? Yeah, me neither.",
            link: "../articles/article.html?article=../articles/building_git_from_scratch_in_python.md"
        },
        {
            title: "Implementing Go from Scratch in Python",
            excerpt: "We're building AlphaGo. The first step, implement Go.",
            link: "../articles/article.html?article=../articles/implementing_go_from_scratch_in_python.md"
        },
        // Add more articles as needed
    ];

    const articlesList = document.getElementById('articles-list');

    articles.forEach(article => {
        const articleCard = document.createElement('div');
        articleCard.classList.add('article-card');

        const articleTitle = document.createElement('h3');
        articleTitle.classList.add('article-title');
        articleTitle.textContent = article.title;

        const articleExcerpt = document.createElement('p');
        articleExcerpt.classList.add('article-excerpt');
        articleExcerpt.textContent = article.excerpt;

        const readMoreLink = document.createElement('a');
        readMoreLink.classList.add('read-more');
        readMoreLink.href = article.link; // Set the link to the article's link
        readMoreLink.textContent = 'Read More';

        articleCard.appendChild(articleTitle);
        articleCard.appendChild(articleExcerpt);
        articleCard.appendChild(readMoreLink);

        articlesList.appendChild(articleCard);
    });
});