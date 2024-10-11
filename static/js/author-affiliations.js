document.querySelectorAll('.affiliation').forEach(affiliation => {
    affiliation.addEventListener('mouseover', () => {
        const affiliationClass = affiliation.getAttribute('data-affiliation') + '-author';
        document.querySelectorAll(`.${affiliationClass}`).forEach(author => {
            switch (affiliation.getAttribute('data-affiliation')) {
                case 'SYSU':
                    author.style.color = '#99ff99'; // SYSU 的颜色
                    break;
                case 'AILAB':
                    author.style.color = '#99ccff'; // AILAB 的颜色
                    break;
                case 'SenseTime':
                    author.style.color = '#ff6347'; // SenseTime 的颜色
                    break;
                case 'CUHK':
                    author.style.color = '#ff9500'; // CUHK 的颜色
                    break;
                case 'CUHK-SZ':
                    author.style.color = '#cc99ff'; // CUHK-SZ 的颜色
                    break;
            }
        });
    });

    affiliation.addEventListener('mouseout', () => {
        const affiliationClass = affiliation.getAttribute('data-affiliation') + '-author';
        document.querySelectorAll(`.${affiliationClass}`).forEach(author => {
            author.style.color = ''; // 恢复原来的颜色
        });
    });
});