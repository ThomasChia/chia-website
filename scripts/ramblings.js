document.addEventListener('DOMContentLoaded', () => {
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
        {
            title: "Building the Policy Network",
            excerpt: "We've implemented Go. Now we need to build the policy network.",
            link: "../articles/article.html?article=../articles/building_the_policy_network.md"
        },
        // Add more articles as needed
    ];

    const articlesList = document.getElementById('articles-list');

    articles.forEach(article => {
        const articleCard = document.createElement('div');
        articleCard.classList.add('article-card');

        const articleLink = document.createElement('a');
        articleLink.href = article.link;
        articleLink.classList.add('article-link'); // Add a class for styling

        const articleTitle = document.createElement('h3');
        articleTitle.classList.add('article-title');
        articleTitle.textContent = article.title;

        const articleExcerpt = document.createElement('p');
        articleExcerpt.classList.add('article-excerpt');
        articleExcerpt.textContent = article.excerpt;

        const readMoreLink = document.createElement('span');
        readMoreLink.classList.add('read-more');
        readMoreLink.textContent = 'Read More';

        articleLink.appendChild(articleTitle);
        articleLink.appendChild(articleExcerpt);
        articleLink.appendChild(readMoreLink);

        articleCard.appendChild(articleLink);
        articlesList.appendChild(articleCard);
    });

    // Ensure MathJax processes the new content
    MathJax.typeset();
});