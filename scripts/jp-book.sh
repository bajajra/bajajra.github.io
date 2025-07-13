jupyter-book clean machinamasqueradebook/machinamasquerade --all
jupyter-book build machinamasqueradebook/machinamasquerade
rm -rf machinamasquerade/*
cp -r machinamasqueradebook/machinamasquerade/_build/html/* machinamasquerade/
