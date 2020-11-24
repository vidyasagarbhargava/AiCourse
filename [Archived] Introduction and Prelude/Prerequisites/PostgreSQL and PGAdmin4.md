# Installing PostgreSQL

Let's install our database service PostgreSQL, and a  PGAdmin4

Firstly

[download PostgreSQL](https://www.enterprisedb.com/downloads/postgres-postgresql-downloads)

and

[download PGAdmin4](https://www.pgadmin.org/download/)

Search for the pgadmin program on your computer and run it. 
It should ask you to set a master password.

Now you should be able to run ```psql``` from your terminal, which will change your prompt.

If you get ```command not found```, it means that your system is not looking in the right place for psql. You need to add the location of this executeable to your path. If you're on windows, follow [this](https://sqlbackupandftp.com/blog/setting-windows-path-for-postgres-tools) guide.